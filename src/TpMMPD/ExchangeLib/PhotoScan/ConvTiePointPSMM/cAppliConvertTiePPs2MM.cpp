#include "ConvertTiePPs2MM.h"
/*===================================================*/
cHomolPS::cHomolPS(	std::string aImgName,
                    int aId,
                    Pt2dr aCoord,
                    int aIdImg
                  ):
    mImgName(aImgName),
    mId(aId),
    mCoord(aCoord),
    mIdImg (aIdImg)
{}
/*===================================================*/

cOneImg::cOneImg(string aImgName):
    mImgName(aImgName),
    mVIsIdExist (1000000, false), //initializer avec 2 milion items ? trop con
    mVCoor (1000000, Pt2dr(0,0)),
    mIdMax (0)
{}
/*===================================================*/

cAppliConvertTiePPs2MM::cAppliConvertTiePPs2MM():
    mIdMaxGlobal (0)
{}

bool cAppliConvertTiePPs2MM::readPSTxtFile(string aPSHomol, vector<cHomolPS*> & VHomolPS)
{
    std::string aImCurrent = " ";
    int counter = 0;
    //read input file : file format to read :
    //  Name Im                 Id Pt        Coord X         Coord Y
    //D0003736.JPG               0           7.997          -1.199
    //File trié par Name Im
    ifstream aFichier(aPSHomol.c_str());
    if(aFichier)
    {
        std::string aLine;
        cOneImg * aImgUnique = NULL;
        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');

            if(aLine.size() != 0)
            {
                //parse
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aName = strtok(aBuffer," ");
                char *aId = strtok( NULL, " " );
                char *aI = strtok( NULL, " " );
                char *aJ = strtok( NULL, " " );
                Pt2dr aC(atof(aI),atof(aJ));
                //creat item PS
                int aIdint = atoi(aId);
//                cHomolPS * aItemPS = new cHomolPS( aName,
//                                                   aIdint,
//                                                   aC
//                                                 );
//                VHomolPS.push_back(aItemPS);
                //Get name Im Unique (uniquement pour fichier trie par name Im)
                if (strcmp(aImCurrent.c_str() , aName.c_str()))
                {
                    aImCurrent = aName;
                    aImgUnique = new cOneImg(aName);
                    mImgUnique.push_back(aImgUnique);
                }
                if (aImgUnique != NULL)
                {
                    aImgUnique->VIsIdExist()[aIdint] = true;
                    aImgUnique->VCoor()[aIdint] = aC;
                    if (aIdint >= aImgUnique->IdMax()) aImgUnique->IdMax() = aIdint;
                }
                counter++;
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
        return false;
    }
    //==== Affichier Result Import Homol PS =====//
    cout<<"Nb Imgs Uniq : " << mImgUnique.size()<< endl;
    cout<<"Nb Item : " << counter<< endl;
    cout<<" ++ => Import PS Homol Finish"  <<endl;
    //cout<<endl<<"Result Import : "<<endl;
    for (uint aKImg=0; aKImg<mImgUnique.size(); aKImg++)
    {
        //cout<<" ++ Img : "<<mImgUnique[aKImg]->ImgName()<<endl;
        //cout<<" ++ IdMax : "<<mImgUnique[aKImg]->IdMax()<<endl;
        if (mImgUnique[aKImg]->IdMax() >= mIdMaxGlobal)
        {
            mIdMaxGlobal = mImgUnique[aKImg]->IdMax();
        }
    }
    cout<<endl<<" ** IdMaxGlobal = "<<mIdMaxGlobal<<endl;
    //============================================//
    for (uint aKImg=0; aKImg<mImgUnique.size(); aKImg++)
    {
       mImgUnique[aKImg]->VIsIdExist().resize(mIdMaxGlobal+1);
       mImgUnique[aKImg]->VCoor().resize(mIdMaxGlobal+1);
    }
    return true;
}

int cAppliConvertTiePPs2MM::getId (int aId, vector<cHomolPS*> & aVRItem)
{
    int counter = 0;
    for (uint aKImg=0; aKImg < mImgUnique.size(); aKImg++)
    {
        if (mImgUnique[aKImg]->VIsIdExist()[aId])
        {
            cHomolPS * aItem = new cHomolPS(mImgUnique[aKImg]->ImgName(), aId,  mImgUnique[aKImg]->VCoor()[aId], aKImg);
            aVRItem.push_back(aItem);
            counter++;
        }
    }
    return counter;
}

void cAppliConvertTiePPs2MM::initAllPackHomol(vector<cOneImg*> VImg)
{
    cout<<"Init Pack Homol ...";
    for (uint aKImg=0; aKImg<VImg.size(); aKImg++)
    {
        for (uint aKImg=0; aKImg<VImg.size(); aKImg++)
        {
            ElPackHomologue * aPack = new ElPackHomologue();
            VImg[aKImg]->Pack().push_back(aPack);
        }
    }
    cout<<"done"<<endl;
}

void cAppliConvertTiePPs2MM::addToHomol(vector<cHomolPS*> aVItemHaveSameId, Pt2dr aCPS, Pt2dr aSizePS)
{
    for (uint i=0; i<aVItemHaveSameId.size(); i++)
    {
        for (uint j=i+1; j<aVItemHaveSameId.size(); j++)
        {
            int aIdImg1 = aVItemHaveSameId[i]->IdImg();
            int aIdImg2 = aVItemHaveSameId[j]->IdImg();
            Pt2dr aCoorImg1 = aVItemHaveSameId[i]->Coord();
            Pt2dr aCoorImg2 = aVItemHaveSameId[j]->Coord();
            aCoorImg1.x = aCPS.x+(aCoorImg1.x/aSizePS.x);
            aCoorImg1.y = aCPS.y - (aCoorImg1.y/aSizePS.y);
            aCoorImg2.x = aCPS.x+(aCoorImg2.x/aSizePS.x);
            aCoorImg2.y = aCPS.y - (aCoorImg2.y/aSizePS.y);
            ElCplePtsHomologues aCpl (aCoorImg1, aCoorImg2);
            mImgUnique[aIdImg1]->Pack()[aIdImg2]->Cple_Add(aCpl);
        }
    }
}

void cAppliConvertTiePPs2MM::writeToDisk(string aOut, bool a2W, string mDir)
{
    std::string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                    +  aOut
                    +  std::string("@")
                    +  std::string("dat");
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(mDir);
    for (uint i=0; i<mImgUnique.size(); i++)
    {
        if (mImgUnique.size()>100)
        {
            if (i %  (mImgUnique.size()/100) == 0)
                 cout<<"   ++ ["<<(i*100.0/mImgUnique.size())<<" %] - write"<<endl;
        }
        vector<ElPackHomologue*> aVPack = mImgUnique[i]->Pack();
        for (uint j=0; j<aVPack.size(); j++)
        {
            std::string clePack = aICNM->Assoc1To2(
                                                    aKHOutDat,
                                                    mImgUnique[i]->ImgName(),
                                                    mImgUnique[j]->ImgName(),
                                                    true
                                                  );
            if (aVPack[j]->size()>0)
                aVPack[j]->StdPutInFile(clePack);
        }
    }
}
