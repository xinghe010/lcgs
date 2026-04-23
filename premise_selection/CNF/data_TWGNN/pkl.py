import pickle

def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == "__main__":

    file_path = "./programs.pkl"
    try:
        loaded_data = load_pkl_file(file_path)
        print("Data loaded successfully:")
        print(loaded_data)
    except Exception as e:
        print(f"Error loading file: {e}")
