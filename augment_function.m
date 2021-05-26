d = dir('data\Aerial_Final_scaled');
for i = 3:1:length(d)
    if(strcmp(d(i).name,'NYC')||strcmp(d(i).name,'ROME')||strcmp(d(i).name,'SF'))
        path1 = 'data\Aerial_Final_scaled';
        save_path1 = 'data\Aerial_Final_scaled_aug';
        %path1
        %save_path1
        subdir = dir(path1);
        
        path2 = strcat(strcat(strcat(path1,'\'),subdir(i).name),'\*.png');
        path3 = strcat(strcat(strcat(path1,'\'),subdir(i).name),'\');
        save_path = strcat(strcat(strcat(save_path1,'\'),subdir(i).name),'\');
        %path2
        %path3
        %save_path
        filelist = dir(path2);
        for k = 1:1:length(filelist)
            I = imread(strcat(path3,filelist(k).name));
            for j = 1:1:15
                I2 = imrotate(I, randi([0,360]));
                I3 = imresize(I2, 1+(randi([50,75])/100));
                I_new = crop_function(I3);
                str = extractBefore(filelist(k).name,'.png');
                filename = strcat(strcat(strcat(str,'_'),num2str(j)),'.png');
                imwrite(I_new,strcat(save_path,filename),'png');
            end
            fprintf(strcat(strcat(strcat('Completed ',path3),' '),'\n'));
        end
        fprintf(strcat(strcat(strcat('Completed ',d(i).name),' '),'\n'));
    end
    %break;
    
end

