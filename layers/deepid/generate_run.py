data_path = '/media/XiangyuZhu_Data/database/webface/batches/'
save_path='model/'
test_range='101-104'
train_range='1-100'
layer_def='layer-define-webface.cfg' 
layer_params='layer-params-webface.cfg' 
data_provider='webface' 
test_freq='13'

part_num = 1

CHANNEL_LIST = [3]

for i in range(part_num):
    nChannels = CHANNEL_LIST[i]
    part_name = 'part%d'%(i+1)
    name = 'run_' + part_name + '.bat'     
    fid = open(name,'wt')
    data_path_part = data_path + part_name + '/'
    save_path_part = save_path + part_name + '/'
    
    command = \
    'python ../../convnet.py ' + \
    '--data-path=' + data_path_part + ' ' + \
    '--data-path-test=' + data_path_part + ' ' + \
    '--save-path=' + save_path_part + ' ' + \
    '--test-range=' + test_range + ' ' + \
    '--train-range=' + train_range + ' ' + \
    '--layer-def=' + layer_def  + ' ' + \
    '--layer-params=' + layer_params + ' ' + \
    '--test-freq=' + test_freq + ' '
    
    if(nChannels == 1):
        command = command + '--data-provider=webface1'
        
    if(nChannels == 3):
        command = command + '--data-provider=webface3'
        
    
    
    fid.write(command)
    fid.close()
    
    
    
    
    
    
    