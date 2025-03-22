import torch

def main():
    file_path = "pldm_envs/diverse_maze/presaved_datasets/40maps/data.p"
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    print(type(data))
    print(len(data))
    
    for in_data in data:
        print(type(in_data))
        
        for k in in_data.keys(): 
            print(f"{k}: type = {type(in_data[k])}", end='')
            if hasattr(in_data[k], "shape"):
                print(f", shape = {in_data[k].shape}")
            else:
                print(f", value = {in_data[k]}")  # int や str などは値を表示

        break  # 最初の1個だけ表示して終了

if __name__ == '__main__':
    main()
