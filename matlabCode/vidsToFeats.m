function vidsToFeats(retrieveHandImagesFrom, toFolder)
    signFolds = getFolderList(retrieveHandImagesFrom, false, false);
    signFolds = sortFolderNames(signFolds, '');
    foldCnt = length(signFolds);
    for sID=1:foldCnt
        %different signs
        videoFolds = getFolderList([retrieveHandImagesFrom filesep signFolds{sID}], false, false);
        videoFolds = sortFolderNames(videoFolds, 'v');
        vidCnt = length(videoFolds);
        for i=1:vidCnt
            %different videos
            [fileNames, imCnt] = getFileList([retrieveHandImagesFrom filesep signFolds{sID} filesep videoFolds{i}], '.png');
            fileNames = sortrows(fileNames);
            vidFeat = [];
            for imID = 1:imCnt
                a = imread([retrieveHandImagesFrom filesep signFolds{sID} filesep videoFolds{i} filesep fileNames{imID}]);
                f = extFt(a);
                vidFeat = [vidFeat;f];
            end
            %save vidFeat to somewhere
            vidFeatFileName = [toFolder filesep 'videoFeats_s' num2str(sID,'%03d') '_v' num2str(i,'%03d') '.txt'];
            saveMatToFile(vidFeatFileName, vidFeat);            
        end
    end
end

function f = extFt(a)
    f = TPLBP(a);
    f = f(:);
    f = f';
end

function saveMatToFile(matFileName, M)
    fid = fopen(matFileName, 'w');
    if fid==-1, error('Cannot open file: %s', matFileName); end
    [rowCnt, colCnt]  = size(M);
    for rid = 1:rowCnt
        for cid = 1:colCnt
            fprintf(fid, '%f ',  M(rid, cid));
        end 
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function foldNamesSorted = sortFolderNames(foldNames, removeChar)
    foldCnt = size(foldNames,1);
    foldIntIDs = zeros(foldCnt,1);
    for i=1:foldCnt
        foldIntIDs(i) = str2double(strrep(foldNames{i},removeChar,''));
    end
    [sortedIDs, idx] = sort(foldIntIDs);
    foldNamesSorted = foldNames(idx);
end