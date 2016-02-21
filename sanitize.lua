require 'torch'

local sanitize = function (net)
    local list = net:listModules()
    for _,val in ipairs(list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if (name == 'output' or name == 'gradInput') then
                val[name] = field.new()
            end
        end
    end
end

return sanitize
