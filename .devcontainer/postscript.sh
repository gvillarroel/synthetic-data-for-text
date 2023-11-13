pip install --upgrade pip
#pip install --user -r requirements.txt

apt update -y
apt upgrade -y
apt upgrade perl -y
apt install pandoc -y
apt install chktex -y

apt install texlive -y
apt install texlive-latex-recommended -y
apt install texlive-extra-utils -y
apt install texlive-latex-extra -y
apt install latexmk -y
apt install tex-common -y
apt install texlive-lang-spanish -y
apt install python3-pygments -y
apt install inkscape -y
#pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 -y
pip install  git+https://github.com/huggingface/peft.git -y

apt-get install texlive-fonts-extra -y
apt-get install -y texlive-bibtex-extra biber

pip install dvc

#(cd notebooks & ln -s ../syntheticml .)