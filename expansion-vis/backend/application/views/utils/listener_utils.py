import os
import re

def get_option_ids(option, class_folder):
    pattern = option + r'(\d+)_'
    ids = set()
    filenames = os.listdir(class_folder)
    # Iterate over filenames and extract IDs that match the pattern
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            id = int(match.group(1))
            ids.add(id)
    return ids

# Retrieve the numeric IDs of images in a specified path. For example, if the folder contains files named like epoch0_split_0_expanded_68.png, 
# we want to capture the number after "epoch".
def get_option_image_ids(option, root_dir):
    image_ids_list = [] 
    class_folders = os.listdir(root_dir)
    class_folders = [os.path.join(root_dir, class_folder) for class_folder in class_folders]
    for class_folder in class_folders:
        ids = get_option_ids(option, class_folder)
        image_ids_list.append(ids)  

    # Check if all subfolder sets are identical
    if all(s == image_ids_list[0] for s in image_ids_list):
        max_set = max(image_ids_list, key=len)
        return max_set
    else:
        # If sets differ, return the set with the minimum number of elements
        min_set = min(image_ids_list, key=len)
        return min_set

# Process images in the specified path, where images start with "epoch" or "hour" followed by a number.
# Save these numbers and return a boolean indicating whether the set of numbers has changed.
class ImageProcessor:
    def __init__(self, directory):
        # Initialize the old set of numbers to an empty set
        self.old_epoch_numbers = get_option_image_ids('epoch', directory)
        self.old_hour_numbers = get_option_image_ids('hour', directory)

    def update_epoch_images(self, directory):
        new_numbers = get_option_image_ids('epoch', directory)
        has_changed = False
        # Compare the new set of numbers with the old set of numbers
        diff_number = None
        if new_numbers != self.old_epoch_numbers:
            # If the sets are different, update the old set and return True
            diff_set = new_numbers - self.old_epoch_numbers
            diff_number = list(diff_set)[0] if diff_set else None
            self.old_epoch_numbers = new_numbers
            has_changed = True
        # Return the number that differs between the two sets
        return has_changed, diff_number
    
    def update_hour_images(self, directory):
        new_numbers = get_option_image_ids('hour', directory)
        has_changed = False
        # Compare the new set of numbers with the old set of numbers
        diff_number = None
        if new_numbers != self.old_hour_numbers:
            # If the sets are different, update the old set and return True
            diff_set = new_numbers - self.old_hour_numbers
            diff_number = list(diff_set)[0] if diff_set else None
            self.old_hour_numbers = new_numbers
            has_changed = True
        # Return the number that differs between the two sets
        return has_changed, diff_number
