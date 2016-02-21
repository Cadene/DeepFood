require 'image'

timer = torch.Timer()

path2train = '/home/cadene/data/UPMC_Food101_224_augmented/train'

print("\n# mean", timer:time().real)

mean = torch.zeros(3, 224, 224)
n = 0

for folder in paths.iterdirs(path2train) do
    path2folder = path2train..'/'..folder
    print(folder, timer:time().real)
    for imname in paths.iterfiles(path2folder) do
        -- print(n.." - "..imname)
        img = image.load(path2folder.."/"..imname)
        if img:size(1) == 1 then
            tmp = torch.zeros(3,224,224)
            tmp[1] = img:clone()
            tmp[2] = img:clone()
            tmp[3] = img:clone()
            img = tmp
        end
        mean:add(img)
        n = n + 1
    end
end

mean = mean / n
image.save(path2train..'/../mean.jpg', mean)

-- STD

print("\n# std", timer:time().real)

std = torch.zeros(3, 224, 224)
n = 0

for folder in paths.iterdirs(path2train) do
    path2folder = path2train..'/'..folder
    print(folder, timer:time().real)
    for imname in paths.iterfiles(path2folder) do
        -- print(n.." - "..imname)
        img = image.load(path2folder.."/"..imname)
        if img:size(1) == 1 then
            tmp = torch.zeros(3,224,224)
            tmp[1] = img:clone()
            tmp[2] = img:clone()
            tmp[3] = img:clone()
            img = tmp
        end
        std:add((img - mean):pow(2))
        n = n + 1
    end
end

std = (std / n):sqrt()
image.save(path2train..'/../std.jpg', std)

print('Time elapsed: ' .. timer:time().real .. ' seconds')