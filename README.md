pip3 install -r requirements.txt

conda activate mandelbrotnn

python3 train_image.py

git init
git add .
git commit -m "adding the main code thats now compatible with cpu only (macos)"
git remote add origin https://github.com/Isaak-C-Augustus/cool_ahhhhhh_ML_image_learning_effect.git
git push -u -f origin main

cd ML_learns_images