# Import libraries
import gdown
import tarfile

def main():
    # Download file
    url = 'https://drive.google.com/u/4/uc?id=16WD0td1f5gx4yIIDkWWSTb-oZcezI1CU&export=download'
    output = './AIA171_Miniset_BW.tar.gz'
    gdown.download(url, output, quiet=False)

    # Get tar file
    tar = tarfile.open(output, 'r:gz')
    
    # extracting file
    tar.extractall()


if __name__ == '__main__':
    main()
