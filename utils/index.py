
def get_hash(array):
    hash = {}
    for screen in array:
        for key, value in screen.items():
            if key not in hash:
                if key == "component":
                    hash[key] = {}
                else: hash[key] = []
                
            if key == "component":
                hash[key][screen["name"]] = value
            else:
                hash[key].append(value) 
    return hash