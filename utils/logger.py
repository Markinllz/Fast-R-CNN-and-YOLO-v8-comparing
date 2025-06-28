import logging

def setup_logger():
    logger = logging.getLogger("YOLOv8")


    if logger.handlers:
        return logger
    

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)


    logger.addHandler(console_handler)


    file_handler = logging.FileHandler('training.log')

    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)
    


    return logger


logger = setup_logger()


