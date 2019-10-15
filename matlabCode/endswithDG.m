function endsWithBool = endswithDG(str,pattern)
    if isempty(pattern)
        endsWithBool = true;
        return
    end
    endsWithBool = false;
    len01_str = length(str);
    len02_pat = length(pattern);
    if len02_pat>len01_str
        return % already false
    end
    initChar = len01_str-len02_pat + 1 ;
    strLastPart = str(initChar:end);
    endsWithBool = strcmpi(pattern, strLastPart);
end