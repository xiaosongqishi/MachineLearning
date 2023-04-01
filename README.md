# MachineLearning
MachineLearning

本项目包含了子模块

如果您对子模块进行了更改，并希望将更改一并提交到您的远程仓库，可以按照以下步骤操作：

1. 提交子模块的更改并push到子模块的远程仓库：

   ```
   cd UDACITY/ud120-projects
   git add .
   git commit -m "Update submodule"
   git push origin main
   ```

2. 返回到主项目的根目录，将子模块提交的更改添加到暂存区，并提交更改：

   ```
   git add .
   git commit -m "Update submodule"
   ```

3. 将整个项目push到您的远程仓库：

   ```
   git push origin main
   ```

注意：当您在主项目中修改了子模块，并提交到主项目的远程仓库时，其他人可能无法直接拉取并使用该更改。他们需要执行 `git submodule update` 命令，以获取最新的子模块更改。
