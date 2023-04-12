import json,logging,hashlib,os

logger = logging.getLogger(__name__) 

class NestedDict(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

class Parameters:
    def __init__(self, json_file_path):
        sha256 = hashlib.sha256()
        with open(json_file_path, "r") as file:
            self.data = json.load(file, object_hook=NestedDict)
            while True:
                data = file.read(65536)
                if not data:
                    break
                sha256.update(data)
            
        self.hash = sha256.hexdigest()[:6]
        self.agent = self.data.agent
        self.params = self.data.params
        self.iterations = self.data.iterations

        os.makedirs("./results/original", exist_ok=True)
        with open(f'./results/original/{self.hash}.json','w') as file:
            file.write(json.dumps({
                'agent': self.agent,
                'hash': self.hash,
                'params': self.params,
                'iterations': self.iterations
            }))
        if os.path.isfile(f'../../results/original/{self.hash}.json'):
            logger.info("Copy of input file created successfully. Parameters Enabled!") 
        else:
            logger.error("Copy of input file not found. Double check before editing input file!") 


    def update_param(self, param_path, new_value, index=None):
        keys = param_path
        data = self.params
        for key in keys[:-1]:
            data = data[key]
        if index is not None and isinstance(data[keys[-1]], list):
            data[keys[-1]][index] = new_value
        else:
            data[keys[-1]] = new_value

    def save_to_json(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def format_output(self) -> NestedDict:
        output = self.data.pop('iterations')
        output.parent_hash = self.hash
        return output
    
def get_nested_value(data, keys, index=None):
        for key in keys:
            data = data[key]
        if index is not None and isinstance(data, list):
            return data[index]
        return data