function [fileNames, numOfFiles] = getFileList(mainFolderName, endsWithStr, startsWith, dispInfo)
    if ~exist('endsWithStr','var') || isempty(endsWithStr)
        endsWithStr = '';
    end
    if ~exist('startsWith','var') || isempty(startsWith)
        startsWith = '';
    end
    if ~exist('dispInfo','var') || isempty(dispInfo)
        dispInfo = false;
    end
    if mainFolderName(end) ~= filesep
        mainFolderName = [mainFolderName filesep];
    end
    
    % Get a list of all files and folders in this folder.
    allFiles = dir(mainFolderName);
    % Get a logical vector that tells which is a directory.
    dirFlags = [allFiles.isdir];
    % Extract only those that are directories.
    fileNames = allFiles(~dirFlags);
    % Extract only names of those directories.
    fileNames = {fileNames.name}';
    % Print folder names to command window.
    k = 1;
    numOfFiles = size(fileNames,1);
    while k<=numOfFiles
        curName = fileNames{k};
        %I wanna know if it meets the endsWith criteria
        %I wanna know if it meets the startsWith criteria
        meetsEndWithCriteria = endswithDG(curName,endsWithStr);
        meetsStrWithCriteria = startswithDG(curName,startsWith);
        if meetsStrWithCriteria && meetsEndWithCriteria
            if dispInfo
                disp(['<' curName '> is added to fileNames list' ]);
            end            
            k = k + 1;
        else
            if dispInfo
                if (~meetsStrWithCriteria) && (~meetsEndWithCriteria)
                    disp(['Removing ' curName ' from the list because it doesnt start with <' startsWith '> and end with <' endsWithStr '>' ]);
                elseif meetsStrWithCriteria && (~meetsEndWithCriteria)
                    disp(['Removing ' curName ' from the list because it doesnt end with <' endsWithStr '>' ]);
                elseif (~meetsStrWithCriteria) && meetsEndWithCriteria
                    disp(['Removing ' curName ' from the list because it doesnt start with <' startsWith '>' ]);
                else
                    error('shouldnt be here');
                end
            end
            fileNames{k} = fileNames{numOfFiles};
            numOfFiles = numOfFiles - 1;
        end
    end
    fileNames = fileNames(1:numOfFiles);
end