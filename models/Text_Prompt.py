import torch
import clip


def text_prompt_openai_pasta_pool_4part():
    print("Use text prompt openai pasta pool")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:5]))) for pasta_list in paste_text_map2])

        elif ii == 1:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in paste_text_map2])
        elif ii == 2:
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ','.join(pasta_list[2:4]))) for pasta_list in paste_text_map2])
        elif ii == 3:
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in paste_text_map2])
        else:
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:]))) for pasta_list in paste_text_map2])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


def text_prompt_openai_pasta_pool_4part_bert():
    print("Use text prompt openai pasta pool bert")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            input_list = [pasta_list[ii] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 1:
            input_list = [','.join(pasta_list[0:2]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 2:
            input_list = [pasta_list[0] + ','.join(pasta_list[2:4]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 3:
            input_list = [pasta_list[0] + ',' + pasta_list[4] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        else:
            input_list = [pasta_list[0] + ',' + ','.join(pasta_list[5:]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list

    return num_text_aug, text_dict


def text_prompt_openai_random_ucla():
    print("Use text prompt openai synonym random UCLA")

    total_list = []
    for pasta_list in ucla_paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list


def text_prompt_openai_pasta_pool_4part_ucla():
    print("Use text prompt openai pasta pool ucla")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ucla_paste_text_map1])
        elif ii == 1:
            text_dict[ii] = torch.cat(
                [clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 2:
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ','.join(pasta_list[2:4]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 3:
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in ucla_paste_text_map1])
        else:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:]))) for pasta_list in
                                       ucla_paste_text_map1])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict
