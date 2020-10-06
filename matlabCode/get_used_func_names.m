function get_used_func_names(file_name, package_name)
%get_used_func_names Checks the file to find used functions from package
%   Opens the file (file_name)
%   Looks at all usages as " <package_name>.XX(..."
%   Prints out unique strings(function names) with which lines they are at
    file_name = ['/home/doga/GithUBuntU/keyhandshapediscovery/' file_name];
    py_file_name = split(file_name,'/');
    py_file_name = strrep(py_file_name{end},'.py','');
    %package_name = 'modelFuncs';
    
    %0. Load the file content as a single string
    ft = fileread(file_name);
    ft = split(ft, newline);
    
    %1. First find how the package imported
    %import <package_name> as XXX
    expr = ['import ' package_name ' as '];
    listRows = find(cell2mat(cellfun(@(x) contains(x,expr),ft, 'UniformOutput', false)));
    if length(listRows)>1
        error(['There are more than 1 line of ' package_name ' in ' py_file_name '. There shouldnt be.']);
    elseif isempty(listRows)
        expr = ['import ' package_name];
        listRows = find(cell2mat(cellfun(@(x) contains(x,expr),ft, 'UniformOutput', false)));  %#ok<EFIND>
        if isempty(listRows)
            warning([package_name ' not imported in ' py_file_name '.']);
            return
        end
        package_disp_name = package_name;
    else
        line_str = ft{listRows};
        package_disp_name = strtrim(strrep(line_str, expr, ''));
    end
    
    disp([package_name ' is imported in ' py_file_name ' as ' package_disp_name]);
    
    %2. Now search for any function called as " XXX.YY(..."
    %   Do this line by line
    expr = [package_disp_name '.'];
    listRows = find(cell2mat(cellfun(@(x) contains(x,expr),ft, 'UniformOutput', false)));
    foundFuncNames = cell(0,2);
    for i=1:length(listRows)
        l = ft{listRows(i)};
        %find the string between <expr> and <(> 
        first_char = strfind(l, expr) + length(expr);
        last_char = strfind(l(first_char:end), '(') -2 + first_char;
        %disp([l,':',num2str(first_char),' to',num2str(last_char)]);
        %disp(l(first_char:last_char));
        foundFuncNames(size(foundFuncNames,1)+1,:) = {l(first_char:last_char), listRows(i)};
    end
    disp(unique(foundFuncNames(:,1)))
    disp(foundFuncNames)
end

