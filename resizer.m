d = dir('data\Aerial_Final');
for i = 3:1:length(d)
    if(strcmp(d(i).name,'NYC')||strcmp(d(i).name,'ROME')||strcmp(d(i).name,'SF'))
        path1 = 'data\Aerial_Final';
        save_path1 = 'data\Aerial_Final_scaled';
        path1
        save_path1
        subdir = dir(path1);
        for j = 3:1:length(subdir)
            if(~(strcmp(subdir(j).name,'.')) && ~(strcmp(subdir(j).name,'..')))
                path2 = strcat(strcat(strcat(path1,'\'),subdir(j).name),'\*.png');
                path3 = strcat(strcat(strcat(path1,'\'),subdir(j).name),'\');
                save_path = strcat(strcat(strcat(save_path1,'\'),subdir(j).name),'\');
                path2
                path3
                save_path
                filelist = dir(path2);
                for k = 1:1:length(filelist)
                    I = imread(strcat(path3,filelist(k).name));
                    I_new = imresize(I, 0.0625);
                    str = extractBefore(filelist(k).name,'.png');
                    filename = strcat(strcat(strcat(str,'_'),num2str(1)),'.png');
                    imwrite(I_new,strcat(save_path,filename),'png');
                end
                fprintf(strcat(strcat(strcat(strcat('Completed ',d(i).name),' '),subdir(j).name),'\n'))
            end
            %break;
        end
    end
    %break;
end

