pip install --upgrade pip
pip install --user -r requirements.txt

apt update -y
apt upgrade -y
apt upgrade perl -y
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

(cd notebooks & ln -s ../syntheticml .)