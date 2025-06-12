import mss
import pytesseract
from PIL import Image
from typing import Dict, List, Any, Tuple

from holistic_agent import config

# Type alias for the structured data returned by the OCR process.
# Each dictionary in the list represents a recognized text element.
OcrData = List[Dict[str, Any]]


class PerceptionModule:
    """
    Manages the agent's perception of the desktop environment.

    This class is responsible for two primary functions:
    1. Capturing the screen to get a visual representation of the current state.
    2. Performing Optical Character Recognition (OCR) on the captured image
       to extract all visible text and its location.

    The processed data (image and structured text) is then provided to other
    modules, such as the ReasoningModule and the core model, to inform the
    agent's next action.
    """

    def __init__(self) -> None:
        """
        Initializes the PerceptionModule.

        Sets up the screen capture tool (mss) and configures the path to the
        Tesseract executable if it's specified in the project configuration.
        """
        self.screen_capturer: mss.mss = mss.mss()
        if hasattr(config, 'TESSERACT_CMD_PATH') and config.TESSERACT_CMD_PATH:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD_PATH


    def capture_screen(self, monitor_number: int = 1) -> Image.Image:
        """
        Captures a screenshot of the specified monitor.

        Uses the 'mss' library for fast and efficient screen grabbing.

        Args:
            monitor_number (int): The monitor number to capture. Defaults to 1,
                                  which is typically the primary monitor.

        Returns:
            Image.Image: A PIL Image object of the captured screen in RGB format.
        """
        try:
            monitor = self.screen_capturer.monitors[monitor_number]
            sct_img = self.screen_capturer.grab(monitor)
            # Convert to PIL Image
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None


    def perform_ocr(self, image: Image.Image) -> OcrData:
        """
        Performs OCR on the provided image to extract text and its properties.

        This method uses Tesseract via pytesseract to extract detailed information
        about the text on the screen, including the text content, position
        (bounding boxes), and confidence levels. This structured format is
        crucial for the agent to understand GUI layouts.

        Args:
            image (Image.Image): The PIL Image object to process.

        Returns:
            OcrData: A list of dictionaries, where each dictionary represents a
                     recognized word or text block. Each dictionary contains keys
                     such as 'text', 'left', 'top', 'width', 'height', and 'conf'
                     (confidence). Returns an empty list if no text is found.
        """
        try:
            data = pytesseract.image_to_data(image, lang=config.PerceptionConfig.OCR_LANGUAGE, output_type=pytesseract.Output.DICT)
            ocr_results = []
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                if data['level'][i] == 5:  # Level 5 corresponds to a word
                    text = data['text'][i]
                    conf = int(data['conf'][i])
                    if text.strip() and conf > 60: # Basic filtering
                        ocr_results.append({
                            'text': text,
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'conf': conf
                        })
            return ocr_results
        except Exception as e:
            print(f"An error occurred during OCR: {e}")
            return []

    def get_observation(self) -> Dict[str, Any]:
        """
        Generates a complete observation of the current screen state.

        This is the main public method of the PerceptionModule. It orchestrates
        the capture and OCR process to create a comprehensive, multimodal snapshot
        of the environment that can be used by the agent for decision-making.

        Returns:
            Dict[str, Any]: A dictionary containing the complete observation,
                            structured as:
                            {
                                'screenshot': Image.Image,  # The captured screen image
                                'ocr_data': OcrData         # Structured OCR results
                            }
        """
        screenshot = self.capture_screen()
        ocr_data = []
        if screenshot:
            ocr_data = self.perform_ocr(screenshot)
        
        return {"screenshot": screenshot, "ocr_data": ocr_data}