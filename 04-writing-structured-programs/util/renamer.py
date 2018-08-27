import os
import os.path


def rename_files(directory_name, base_filename):
    """
    Rename files in the given directory
    :param directory_name: input directory
    :param base_filename: base string for the new filename
    :return: None
    """
    count = 0
    for file in os.listdir(directory_name):
        count += 1
        try:
            os.rename(os.path.join(directory_name, file),
                      os.path.join(directory_name, base_filename % str(count)))
        except Exception as e:
            print("Error Type: " + str(e))
            print("Count is: " + str(count))


def main():
    folder = './data/'
    base_filename = 'Jennifer_Garner_Zimbio_%s.jpg'
    rename_files(folder, base_filename)


if __name__ == '__main__':
    main()
