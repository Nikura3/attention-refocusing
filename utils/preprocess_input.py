import numpy as np

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d

def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att

def format_box(names, boxes):
    result_name = []
    resultboxes = []
    for i, name in enumerate(names):
        name = remove_numbers(name)
        result_name.append('a ' + name.replace('_',' '))
        if name == 'person': 
            boxes[i] = boxes[i]
        resultboxes.append([boxes[i]])
    return result_name, np.array(resultboxes)