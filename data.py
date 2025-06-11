import pickle
import pandas as pd

file_path = "./final_project_dataset.pkl"

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    if isinstance(data, dict):
        print("The .pkl file contains a dictionary.")
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')  # Adjust orientation as needed
    elif isinstance(data, pd.DataFrame):
        print("The .pkl file contains a pandas DataFrame.")
        df = data
    else:
        print(f"The .pkl file contains an unsupported data type: {type(data)}")
        df = None

    # Display the data in a table format
    if df is not None:
        print("Data successfully loaded into a DataFrame:")
        print("====================================")
        print("Column names in the DataFrame:")
        print(list(df.columns))
        
        
        # Check if the 'poi' column exists
        if 'poi' in df.columns:
            # Filter rows where 'poi' is True
            poi_names = df[df['poi'] == True].index.tolist()
            print(f"\nTotal number of Persons of Interest (POI): {len(poi_names)}")
            print("Names of Persons of Interest (POI):")
            print(poi_names)
        else:
            print("The dataset does not contain a 'poi' column.")

except Exception as e:
    print(f"An error occurred while loading the .pkl file: {e}")