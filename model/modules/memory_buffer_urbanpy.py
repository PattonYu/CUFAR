import torch
from random import shuffle

def reservoir(num_seen_examples: int, buffer_size: int, rand_len: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :rand_len: the length of random list
    :return: the target index if the data is sampled, else None
    """
    rand = torch.randperm(num_seen_examples + 1)[0:rand_len]
    rand_index = [rand[i].item() for i in range(rand.size(0)) if rand[i] < buffer_size]

    if len(rand_index)== 0:
        return None
    else:
        return torch.tensor(rand_index) 

class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, n_tasks, args):
        self.buffer_size = buffer_size
        self.new_buffer_size = int(buffer_size/2)
        self.num_seen_examples = 0
        self.seen_task = 0
        self.n_tasks = n_tasks
        self.args = args
        self.buffer_maps_ext = list()
        self.buffer_c_maps = torch.zeros([buffer_size, args.channels, args.c_map_shape, args.c_map_shape])
        self.buffer_m_maps = torch.zeros([buffer_size, args.channels, args.c_map_shape*2, args.c_map_shape*2])
        self.buffer_f_maps = torch.zeros([buffer_size, args.channels, args.f_map_shape, args.f_map_shape])
        self.buffer_ext = torch.zeros([buffer_size, args.ext_shape])
        
        self.buffer_maps_ext.append(self.buffer_c_maps)
        self.buffer_maps_ext.append(self.buffer_m_maps)
        self.buffer_maps_ext.append(self.buffer_f_maps)
        self.buffer_maps_ext.append(self.buffer_ext)

        self.buffer_c_maps_new_task = None
        self.buffer_m_maps_new_task = None
        self.buffer_f_maps_new_task = None
        self.buffer_ext_new_task = None

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def add_data(self, flows, t):
        c_maps = flows[0].cpu().detach()
        m_maps = flows[1].cpu().detach()
        f_maps = flows[2].cpu().detach()
        ext = flows[3].cpu().detach()
        if t != self.seen_task:
            if t == 2 and self.args.initial_train is True:
                print('='*17, "SAVE Task-1 Buffer", '='*17)
                torch.save(self.buffer_maps_ext, 'buffers/{}_stage1_maps_ext_{}.pt'.format(self.args.dataset, self.buffer_size))
            
            self.seen_task = t
            self.num_seen_examples = 0

            if self.buffer_c_maps_new_task is not None and t != 2:
                rand_list = [i for i in range(self.buffer_size)]
                shuffle(rand_list)
                rand_list = torch.tensor(rand_list[:self.buffer_c_maps_new_task.size(0)])
                
                self.buffer_c_maps[rand_list] = self.buffer_c_maps_new_task[:rand_list.size(0)]
                self.buffer_m_maps[rand_list] = self.buffer_m_maps_new_task[:rand_list.size(0)]
                self.buffer_f_maps[rand_list] = self.buffer_f_maps_new_task[:rand_list.size(0)]
                self.buffer_ext[rand_list] = self.buffer_ext_new_task[:rand_list.size(0)]

                print('='*10,"replace half of M with M_sub",'='*10)
                print('='*14,"size of M_sub:{}".format(rand_list.size(0)),'='*14)

            # # initia new task buffer
            self.buffer_c_maps_new_task = torch.zeros(
                        [int(self.new_buffer_size), self.args.channels, self.args.c_map_shape, self.args.c_map_shape])
            self.buffer_m_maps_new_task = torch.zeros(
                        [int(self.new_buffer_size), self.args.channels, self.args.c_map_shape *2, self.args.c_map_shape *2])
            self.buffer_f_maps_new_task = torch.zeros(
                        [int(self.new_buffer_size), self.args.channels, self.args.f_map_shape, self.args.f_map_shape])
            self.buffer_ext_new_task = torch.zeros([int(self.new_buffer_size), self.args.ext_shape])
        
        if t == self.n_tasks:
            self.buffer_c_maps_new_task = None
            self.buffer_m_maps_new_task = None
            self.buffer_f_maps_new_task = None
            self.buffer_ext_new_task = None
            return

        if t == 1:
            if self.buffer_size >= self.num_seen_examples + c_maps.size(0):
                self.buffer_c_maps[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = c_maps
                self.buffer_m_maps[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = m_maps
                self.buffer_f_maps[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = f_maps
                self.buffer_ext[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = ext
                
                self.num_seen_examples += c_maps.size(0)

            elif self.buffer_size <= self.num_seen_examples:
                # reservoir algorithm add data
                list_index = reservoir(self.num_seen_examples, self.buffer_size, c_maps.size(0))
                if list_index is not None: 
                    self.buffer_c_maps[list_index] = c_maps[:list_index.size(0)]
                    self.buffer_m_maps[list_index] = m_maps[:list_index.size(0)]
                    self.buffer_f_maps[list_index] = f_maps[:list_index.size(0)]
                    self.buffer_ext[list_index] = ext[:list_index.size(0)]
                    self.num_seen_examples += c_maps.size(0)
                else:
                    return
            
            else:
                length = self.buffer_size - self.num_seen_examples
                self.buffer_c_maps[self.num_seen_examples:] = c_maps[:length]
                self.buffer_m_maps[self.num_seen_examples:] = m_maps[:length]
                self.buffer_f_maps[self.num_seen_examples:] = f_maps[:length]
                self.buffer_ext[self.num_seen_examples:] = ext[:length]
                
                self.num_seen_examples += length

        else:
            if self.new_buffer_size >= self.num_seen_examples + c_maps.size(0):
                self.buffer_c_maps_new_task[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = c_maps
                self.buffer_m_maps_new_task[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = m_maps
                self.buffer_f_maps_new_task[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = f_maps
                self.buffer_ext_new_task[self.num_seen_examples:self.num_seen_examples+c_maps.size(0)] = ext
                
                self.num_seen_examples += c_maps.size(0)

            elif self.new_buffer_size <= self.num_seen_examples:
                # reservoir algorithm add data
                list_index = reservoir(self.num_seen_examples, self.new_buffer_size, c_maps.size(0))
                if list_index is not None: 
                    self.buffer_c_maps_new_task[list_index] = c_maps[:list_index.size(0)]
                    self.buffer_m_maps_new_task[list_index] = m_maps[:list_index.size(0)]
                    self.buffer_f_maps_new_task[list_index] = f_maps[:list_index.size(0)]
                    self.buffer_ext_new_task[list_index] = ext[:list_index.size(0)]
                    self.num_seen_examples += c_maps.size(0)
                else:
                    return
            
            else:
                length = self.new_buffer_size - self.num_seen_examples
                self.buffer_c_maps_new_task[self.num_seen_examples:] = c_maps[:length]
                self.buffer_m_maps_new_task[self.num_seen_examples:] = m_maps[:length]
                self.buffer_f_maps_new_task[self.num_seen_examples:] = f_maps[:length]
                self.buffer_ext_new_task[self.num_seen_examples:] = ext[:length]
                
                self.num_seen_examples += length

    def get_data(self, minibatch_size, t):
        index_list = [i for i in range(self.buffer_size)]
        shuffle(index_list)
        index_list = torch.tensor(index_list[0:minibatch_size])

        ret_list = list()
        for i in range(len(self.buffer_maps_ext)):
            ret_list.append(self.buffer_maps_ext[i][index_list].cuda())

        return ret_list