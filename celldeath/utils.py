import os




def create_folder(folder):
    home_dir= os.path.expanduser('~user') 
    if not os.path.exists(home_dir+folder):
        os.makedirs(home_dir+folder)

