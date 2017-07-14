# ASDL2017
Course project for ["Analysing Software using Deep Learning"](https://www.sola.tu-darmstadt.de/index.php?id=13101). See the [project description](https://www.sola.tu-darmstadt.de/fileadmin/user_upload/Group_SOLA/Teaching/summer_2017/ASDL/project_description_20170529.pdf) for details.


## Development
```shell
# INSTALLATION
$ brew install pyenv pyenv-virtualenv
$ pyenv install 3.5.2
$ pyenv virtualenv 3.5.2 asdl
$ pyenv activate 3.5.2/envs/asdl
$ pip install -r requirements.txt
$ pyenv deactivate

# eventually replace ".bash_profile" withe the rc-file of your shell (e.g. .zshrc)
$ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile

# ACTIVATION
$ pyenv activate 3.5.2/envs/asdl
```