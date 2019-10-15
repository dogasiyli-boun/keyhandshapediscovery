function startsWithBool = startswithDG(str,pattern)
    if isempty(pattern)
        startsWithBool = true;
        return
    end
    startsWithBool = false;
    len01_str = length(str);
    len02_pat = length(pattern);
    if len02_pat>len01_str
        return % already false
    end
    strInitPart = str(1:len02_pat);
    startsWithBool = strcmpi(pattern, strInitPart);
end