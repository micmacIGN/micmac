/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "Detector.h"
#include "../../uti_phgrm/NewOri/NewOri.h"
#include "PHO_MI.h"


Detector::Detector( string typeDetector, vector<double> paramDetector,
                    string nameImg,
                    Im2D<unsigned char, int> * img = NULL ,
                    InitOutil * aChain = NULL
                   )
{
    mTypeDetector = typeDetector;
    mParamDetector = paramDetector;
    mChain = aChain;
    if ((img == NULL))
    {
        string aDir;
        if (mChain == NULL)
            aDir = "./";
        else
            aDir = mChain->getPrivmICNM()->Dir();
        Tiff_Im * mPicTiff = new Tiff_Im
                            ( Tiff_Im::StdConvGen(aDir+mNameImg,1,false) );
        TIm2D<U_INT1,INT4> * mPic_TIm2D = new TIm2D<U_INT1,INT4> (mPicTiff->sz());
        ELISE_COPY(mPic_TIm2D->all_pts(), mPicTiff->in(), mPic_TIm2D->out());
        mImg = new Im2D<U_INT1,INT4> (mPic_TIm2D->_the_im);
    }
    else if(img != NULL)
    {
        mImg = img;
        mNameImg = nameImg;
    }
    else
        cout<<"ERROR creat Detector: Il fault avoir (nameImg + mICNM) ou Im2D"<<endl;
}

Detector::Detector( string typeDetector, vector<double> paramDetector,
                    pic * aPic,
                    InitOutil * aChain
                    )
{
    mTypeDetector = typeDetector;
    mParamDetector = paramDetector;
    mNameImg = aPic->getNameImgInStr();
    mChain = aChain;
    mImg = aPic->mPic_Im2D;
}

Detector::Detector( InitOutil * aChain , pic * pic1 , pic * pic2)    //from pack homo Init
{
    mChain = aChain;
    mNameImg = pic1->getNameImgInStr();
    mPic2 = pic2;
    this->mTypeDetector = "HOMOLINIT";
}

vector<Pt2dr> Detector::importFromHomolInit(pic* pic2)
{
    string HomoIn = this->mChain->getPrivmICNM()->Assoc1To2(this->mChain->getPrivMember("mKHIn"),
                                                            mNameImg,
                                                            *pic2->mNameImg,true);
    StdCorrecNameHomol_G(HomoIn,this->mChain->getPrivmICNM()->Dir());
    bool Exist = ELISE_fp::exist_file(HomoIn);
    if (Exist)
    {
        ElPackHomologue apackInit =  ElPackHomologue::FromFile(HomoIn);
        for (ElPackHomologue::const_iterator itP=apackInit.begin(); itP!=apackInit.end() ; itP++)
        {
            this->mPtsInterest.push_back(itP->P1());
        }
    }
    return mPtsInterest;
}

void Detector::getmPtsInterest(vector<Pt2dr> & ptsInteret)
{
    ptsInteret = this->mPtsInterest;
}

void Detector::saveResultToDiskTypeDigeo(string aDir = "./")
{
    if (mPtsInterest.size() == 0)
        cout<<"+++ +++ WARNING +++ +++ : NO POINTS INTEREST TO SAVE"<<endl;
    string outDigeoFile =  mNameImg + "_" + mTypeDetector + ".dat";
    vector<DigeoPoint> points;
    for (uint i=0; i<mPtsInterest.size(); i++)
    {
        DigeoPoint aPt;
        aPt.x = mPtsInterest[i].x;
        aPt.y = mPtsInterest[i].y;
        points.push_back(aPt);
    }
    string aPtsInteretOutDirName;
    if (mChain != NULL)
        aPtsInteretOutDirName=mChain->getPrivmICNM()->Dir()+"PtsInteret/";
    else
        aPtsInteretOutDirName = aDir + "PtsInteret/";
    if(!(ELISE_fp::IsDirectory(aPtsInteretOutDirName)))
        ELISE_fp::MkDir(aPtsInteretOutDirName);
    if ( !DigeoPoint::writeDigeoFile(aPtsInteretOutDirName + outDigeoFile, points))
        ELISE_ERROR_EXIT("failed to write digeo file [" << outDigeoFile << "]");
}

void Detector::saveResultToDiskTypeElHomo(string aDir = "./")
{
    if (mPtsInterest.size() == 0)
        cout<<"+++ +++ WARNING +++ +++ : NO POINTS INTEREST TO SAVE"<<endl;
    string outElHomoFile =  mNameImg + "_" + mTypeDetector + ".dat";
     ElPackHomologue aPackOut;
    for (uint i=0; i<mPtsInterest.size(); i++)
    {
        ElCplePtsHomologues aCplPts(mPtsInterest[i], mPtsInterest[i]);
        aPackOut.Cple_Add(aCplPts);
    }
    string aPtsInteretOutDirName;
    if (mChain != NULL)
        aPtsInteretOutDirName=mChain->getPrivmICNM()->Dir()+"PtsInteret/";
    else
        aPtsInteretOutDirName = aDir + "PtsInteret/";
    if(!(ELISE_fp::IsDirectory(aPtsInteretOutDirName)))
        ELISE_fp::MkDir(aPtsInteretOutDirName);
    aPackOut.StdPutInFile(aPtsInteretOutDirName + outElHomoFile);
}

int Detector::detect(bool useTypeFileDigeo)
{
    int temp=0;
    string outDigeoFile =  mNameImg + "_" + mTypeDetector + ".dat";
    string aPtsInteretOutDirName; string aDir = "./";
    if (mChain != NULL)
        aDir = mChain->getPrivmICNM()->Dir();
    else
        aDir = "./";
    aPtsInteretOutDirName=aDir+"PtsInteret/";
    string aHomoIn = aPtsInteretOutDirName + outDigeoFile;
    bool Exist= ELISE_fp::exist_file(aHomoIn);
    if (mTypeDetector != "HOMOLINIT")
    {
        if (Exist)
        {
            cout<<" Found Pack Pts Interet : "<< aHomoIn<<endl;
            if (useTypeFileDigeo)
                readResultDetectFromFileDigeo(aHomoIn);
            else
                readResultDetectFromFileElHomo(aHomoIn);
        }
        else
        {
            if (mTypeDetector == "FAST")
            {
                if (mParamDetector.size() == 0)
                {cout<<"ERROR : please saisir parameter for FAST detector (dParam)"<<endl;}
                Fast aDetecteur(mParamDetector[0], 3);
                aDetecteur.detect(*mImg, mPtsInterest);
                if (useTypeFileDigeo)
                    this->saveResultToDiskTypeDigeo();
                else
                    this->saveResultToDiskTypeElHomo();
            }
            if (mTypeDetector == "DIGEO")
            {
                string outDigeoFile =  mNameImg + "_DIGEO.dat";
                string command_Digeo = "mm3d Digeo "+mNameImg+" -o "+outDigeoFile;
                temp = system(command_Digeo.c_str());
                readResultDetectFromFileDigeo(outDigeoFile);
                if (useTypeFileDigeo)
                    this->saveResultToDiskTypeDigeo();
                else
                    this->saveResultToDiskTypeElHomo();
                ELISE_fp::RmFileIfExist(outDigeoFile);
            }
        }
    }
    else
    {
        if (Exist)
        {
            string command_rm = "rm "+aHomoIn;
            int temp = system(command_rm.c_str()); cout<<" -- "<<temp<<" -- "<<endl;
            //ELISE_fp::RmFileIfExist(aHomoIn);     //bug par hasard if using this ??? WHY
        }
        this->importFromHomolInit(mPic2);
        if (useTypeFileDigeo)
            this->saveResultToDiskTypeDigeo();
        else
            this->saveResultToDiskTypeElHomo();
    }
    cout<<" "<<mTypeDetector<<" : "<<mPtsInterest.size()<<" Pts in "<<mNameImg<<endl;
    return temp;
}

void Detector::saveToPicTypeVector(pic* aPic)
{
     aPic->mListPtsInterestFAST = mPtsInterest;
}

void Detector::saveToPicTypePackHomol(pic* aPic)
{
    ElPackHomologue packOut;
    for (uint i=0; i<mPtsInterest.size(); i++)
    {
        Pt2dr pts(mPtsInterest[i].x, mPtsInterest[i].y);
        ElCplePtsHomologues aCouple(pts,pts);
        packOut.Cple_Add(aCouple);
    }
    aPic->mPackFAST = packOut;
}



int Detector::readResultDetectFromFileDigeo(string filename)
{
    DigeoPoint fileDigeo;
    vector<DigeoPoint> listPtDigeo;
    cout<<"Read : "<<filename<<endl;
    bool ok = fileDigeo.readDigeoFile(filename, 1,listPtDigeo);
    for (uint i=0; i<listPtDigeo.size(); i++)
    {
        mPtsInterest.push_back(Pt2dr(listPtDigeo[i].x, listPtDigeo[i].y));
    }
    if (!ok)
        cout<<" DIGEO File read error ! "<<endl;
    cout<<"Nb : "<<listPtDigeo.size()<<endl;
    return listPtDigeo.size();
}


int Detector::readResultDetectFromFileElHomo(string filename)
{
    ElPackHomologue aPackIn;
    bool Exist= ELISE_fp::exist_file(filename);
    if (Exist)
    {
        aPackIn =  ElPackHomologue::FromFile(filename);
        cout<<" + Found Pack Homo "<<aPackIn.size()<<" pts"<<endl;
        for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
            mPtsInterest.push_back(itP->P1());
    }
    else
    {
        cout<<" Elhomo File read error ! "<<filename<<endl;
    }
    return mPtsInterest.size();
}



