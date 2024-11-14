import os
from PIL import Image
import argparse
from tqdm import tqdm


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_trimming.py',
    description='Trim the specified area of the image',
    epilog='End of help message',
    usage='python -m ttools/k_trimming.py [img_folder_path] [area_L] [area_R] [area_T] [area_B]',
)
parser.add_argument('img_folder_path', help='Path to the folder containing the images to be trimmed')
parser.add_argument('area_L', type=int, help='Left side of the area to be trimmed')
parser.add_argument('area_R', type=int, help='Right side of the area to be trimmed')
parser.add_argument('area_T', type=int, help='Top side of the area to be trimmed')
parser.add_argument('area_B', type=int, help='Bottom side of the area to be trimmed')
args = parser.parse_args()



#* Define the input and output folders
input_folder = args.img_folder_path
output_folder = os.path.join(os.path.dirname(input_folder), 'screenshot_trimmed')
os.makedirs(output_folder, exist_ok=True)



#* Define the area to be trimmed
#* (left, upper, right, lower) in percent
crop_area_percent = [args.area_L, args.area_T, args.area_R, args.area_B]



#* Process all images in the input folder
for filename in tqdm(os.listdir(input_folder), desc='Processing'):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Open the image
        with Image.open(input_path) as img:
            width, height = img.size
            
            #* Calculate the area to be trimmed
            left = int((crop_area_percent[0] / 100) * width)
            upper = int((crop_area_percent[1] / 100) * height)
            right = int((crop_area_percent[2] / 100) * width)
            lower = int((crop_area_percent[3] / 100) * height)
            
            #* Define the area to be trimmed
            crop_area = (left, upper, right, lower)
            
            #* Trim the image
            cropped_img = img.crop(crop_area)
            #* Save the trimmed image
            cropped_img.save(output_path)
            #print(f'{filename} をトリミングして保存しました。')
