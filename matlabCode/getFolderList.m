function subFolderNames = getFolderList( mainFolderName, dispInfo, addRootPath)
    if ~exist('addRootPath','var') || isempty(addRootPath)
        addRootPath = false;
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
    subFolders = allFiles(dirFlags);
    % Extract only names of those directories.
    subFolderNames = {subFolders.name}';
    % Print folder names to command window.
    k = 1;
    K = size(subFolderNames,1);
    while k<=K
        curName = subFolderNames{k};
        if curName(1)=='.'
            subFolderNames{k} = subFolderNames{K};
            K = K - 1;
        else
            if addRootPath
                subFolderNames{k} = [mainFolderName subFolderNames{k}];
            end
            if dispInfo
                disp(['<' curName '> is added to subFolder list' ]);
            end
            k = k + 1;
        end
    end
    subFolderNames = subFolderNames(1:K);
end

