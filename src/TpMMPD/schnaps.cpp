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

#include "StdAfx.h"
#include <fstream>
#include <algorithm>

/**
 * Schnaps : reduction of homologue points in image geometry
 *    S trict
 *    C hoice of
 *    H omologous
 *    N eglecting 
 *    A ccumulations on
 *    P articular
 *    S pots
 * 
 *
 * Inputs:
 *  - pattern of images
 *  - Homol dir
 *  - minimal number of searching windows in each picture
 *  - new homol dir name
 *
 * Output:
 *  - new homol dir
 *
 * Call example:
 *   mm3d Schnaps ".*.tif" NbWin=100 HomolOut=Homol_mini100
 *
 * Info: jmmuller
 * 
 * */

//#define ReductHomolImage_DEBUG

//----------------------------------------------------------------------------
// PicSize class: only to compute windows size for one given picture size
class cPicSize
{
  public:
    cPicSize
    (
        Pt2di aSz,int aNumWindows
    );
    Pt2di getPicSz(){return mPicSz;}
    Pt2di getWinSz(){return mWinSz;}
    Pt2di getNbWin(){return mNbWin;}

  protected:
    Pt2di mPicSz;
    Pt2di mWinSz;
    Pt2di mNbWin;
  };


cPicSize::cPicSize(Pt2di aSz,int aNumWindows) :
    mPicSz(aSz)
{
    float aXYratio=((float)aSz.x)/aSz.y;
    mNbWin.x=sqrt(aNumWindows)*sqrt(aXYratio)+1;
    mNbWin.y=sqrt(aNumWindows)/sqrt(aXYratio)+1;
    mWinSz.x=((float)mPicSz.x)/mNbWin.x+0.5;
    mWinSz.y=((float)mPicSz.y)/mNbWin.y+0.5;
}

//----------------------------------------------------------------------------
// PointOnPic class: an Homol point with coordinates on several pictures
class cPic;
class cHomol;
class cPointOnPic
{
  public:
    cPointOnPic
    (
        cPic *aPic,Pt2dr aPt,cHomol* aHomol
    );
    cPic* getPic(){return mPic;}
    Pt2dr getPt(){return mPt;}
    cHomol* getHomol(){return mHomol;}
    void setHomol(cHomol* aHomol){mHomol=aHomol;}
    void print();
  protected:
    cPic* mPic;
    Pt2dr mPt;
    cHomol* mHomol;//the Homol it cames from
    
};

//----------------------------------------------------------------------------
// Homol class: an Homol point with coordinates on several pictures
class cHomol
{
  public:
    cHomol();
    ~cHomol();
    int getId(){return mId;}
    std::vector<cPointOnPic*> & getPointOnPics(){return mPointOnPics;}
    bool alreadyIn(cPic * aPic, Pt2dr aPt);
    void add(cPic * aPic, Pt2dr aPt);
    void add(cHomol *aHomol);
    void print();
    bool isBad(){return mBad;}
    void setBad(){mBad=true;}
    cPointOnPic* getPointOnPic(cPic * aPic);
  protected:
    int mId;//unique id
    static int mHomolCounter;
    std::vector<cPointOnPic*> mPointOnPics;
    bool mBad;
};
int cHomol::mHomolCounter=0;

cHomol::cHomol() : mBad(false)
{
    mId=mHomolCounter;
    mHomolCounter++;
}
cHomol::~cHomol()
{
    for(unsigned int i=0;i<mPointOnPics.size();i++)
        delete mPointOnPics[i];
    mPointOnPics.clear();
}
void cHomol::print()
{
    std::cout<<"  cHomol "<<mId<<" ("<<this<<")  multi: "<<mPointOnPics.size()<<":"<<std::endl;
    for(unsigned int i=0;i<mPointOnPics.size();i++)
      mPointOnPics[i]->print();
}
//----------------------------------------------------------------------------
// Pic class: a picture with its homol points
class cPic
{
  public:
    cPic(std::string aName,std::vector<cPicSize> & allSizes,int aNumWindows);
    cPic(cPic* aPic);
    std::string getName(){return mName;}
    cPicSize * getPicSize(){return mPicSize;}
    std::list<cPointOnPic*> *getAllPointsOnPic(){return &mAllPointsOnPic;}
    std::list<cPointOnPic*> *getAllSelectedPointsOnPic(){return &mAllSelectedPointsOnPic;}
    bool removeHomolPoint(cPointOnPic* aPointOnPic);
    void printHomols();
    bool addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic);
    float getPercentWinUsed();
    void incNbWinUsed(){mNbWinUsed++;}
  protected:
    std::string mName;
    cPicSize * mPicSize;
    std::list<cPointOnPic*> mAllPointsOnPic;
    std::list<cPointOnPic*> mAllSelectedPointsOnPic;
    int mNbWinUsed;
};

cPic::cPic(std::string aName,std::vector<cPicSize> & allSizes,int aNumWindows) :
    mName(aName),mPicSize(0),mNbWinUsed(0)
{
    Tiff_Im aPic(aName.c_str());
    Pt2di aPicSize=aPic.sz();
    bool found=false;
    for (unsigned int i=0;i<allSizes.size();i++)
      if (allSizes[i].getPicSz()==aPicSize)
      {
        found=true;
        mPicSize=&allSizes[i];
        break;
      }
    if (!found)
    {
      allSizes.push_back(cPicSize(aPicSize,aNumWindows));
      mPicSize=&allSizes.back();
    }
}

cPic::cPic(cPic * aPic) :
    mName(aPic->mName),mPicSize(aPic->mPicSize)
{
}

void cPic::printHomols()
{
    std::cout<<mName<<" homols:\n";
    std::list<cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=mAllPointsOnPic.begin();itPointOnPic!=mAllPointsOnPic.end();++itPointOnPic)
    {
        std::cout<<(*itPointOnPic)<<" "<<(*itPointOnPic)->getPt()<<" "<<(*itPointOnPic)->getHomol()
                 <<" multi "<<(*itPointOnPic)->getHomol()->getPointOnPics().size()<<std::endl;
    }
}

bool cPic::addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic)
{
    std::list<cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=mAllSelectedPointsOnPic.begin();itPointOnPic!=mAllSelectedPointsOnPic.end();++itPointOnPic)
    {
        if ((*itPointOnPic)==aPointOnPic)
            return false;
    }
    mAllSelectedPointsOnPic.push_back(aPointOnPic);
    return true;
}

float cPic::getPercentWinUsed()
{
    return 100.0*((float)mNbWinUsed)/(mPicSize->getNbWin().x*mPicSize->getNbWin().y);
}

//----------------------------------------------------------------------------
cPointOnPic::cPointOnPic(cPic *aPic,Pt2dr aPt,cHomol* aHomol) :
    mPic(aPic),mPt(aPt),mHomol(aHomol)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"New cPointOnPic "<<this<<" for Homol "<<mHomol->getId()<<"   homol "<<mHomol<<std::endl;
#endif
}

void cPointOnPic::print()
{
    std::cout<<"     cPointOnPic "<<this<<" on "<<mPic->getName()<<": "<<mPt<<"   homol "<<mHomol<<std::endl;
}

cPointOnPic* cHomol::getPointOnPic(cPic * aPic)
{
    for (unsigned int i=0;i<mPointOnPics.size();i++)
        if (mPointOnPics[i]->getPic()==aPic)
            return mPointOnPics[i];
    return 0;
}


bool cHomol::alreadyIn(cPic * aPic, Pt2dr aPt)
{
    //std::cout<<"Homol "<<mId<<"\n";
    //std::cout<<"Look for pic "<<aPic->getName()<<"\n";
    for (unsigned int i=0;i<mPointOnPics.size();i++)
    {
        //std::cout<<"is "<<mPointOnPics[i]->getPic()->getName()<<" ?  ";
        if (mPointOnPics[i]->getPic()==aPic)
        {
            //std::cout<<"yes... ";
            if ((fabs(mPointOnPics[i]->getPt().x-aPt.x)<0.1)&&
                (fabs(mPointOnPics[i]->getPt().y-aPt.y)<0.1))
            {
                //std::cout<<"found\n";
                return true;
            }
            //std::cout<<"other point\n";
            return false;
        }
        //std::cout<<"\n";
    }
    return false;
}

void cHomol::add(cPic * aPic, Pt2dr aPt)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"add PointOnPic "<<aPic->getName()<<" "<<aPt<<" on Homol "<<mId<<" \n";
#endif
    mPointOnPics.push_back(new cPointOnPic(aPic,aPt,this));
    aPic->getAllPointsOnPic()->push_back(mPointOnPics.back());
}

void cHomol::add(cHomol *aHomol)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"Merge Homol "<<mId<<" and "<<aHomol->mId<<"\n";
#endif
    for (unsigned int i=0;i<aHomol->getPointOnPics().size();i++)
    {
        mPointOnPics.push_back(aHomol->getPointOnPics()[i]);
        mPointOnPics.back()->setHomol(this);
        aHomol->getPointOnPics()[i]->getPic()->removeHomolPoint(aHomol->getPointOnPics()[i]);
        aHomol->getPointOnPics()[i]->getPic()->getAllPointsOnPic()->push_back(mPointOnPics.back());
    }
}

bool cPic::removeHomolPoint(cPointOnPic* aPointOnPic)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"cPic::removeHomolPoint"<<std::endl;
#endif
    std::list<cPointOnPic*>::iterator itHomolPoint;
    for (itHomolPoint=mAllPointsOnPic.begin();itHomolPoint!=mAllPointsOnPic.end();++itHomolPoint)
    {
        if ((*itHomolPoint)==aPointOnPic)
        {
            mAllPointsOnPic.erase(itHomolPoint);
            return true;
        }
    }
    return false;
}

bool compareNumberOfHomolPics (cPic* aPic1,cPic* aPic2)
{
    return (aPic1->getAllPointsOnPic()->size()<aPic2->getAllPointsOnPic()->size());
}
//----------------------------------------------------------------------------

int schnaps_main(int argc,char ** argv)
{
    std::string aFullPattern;//pattern of all images
    std::string aInHomolDirName="";//input Homol dir suffix
    std::string aOutHomolDirName="_mini";//output Homol dir suffix
    int aNumWindows=100;//minimal homol points in each picture
    bool ExpTxt=false;//Homol are in dat or txt

    std::cout<<"Schnaps : reduction of homologue points in image geometry\n"
            <<"S trict           \n"
            <<"C hoice of        \n"
            <<"H omologous       \n"
            <<"N eglecting       \n"
            <<"A ccumulations on \n"
            <<"P articular       \n"
            <<"S pots"<<std::endl;

    ElInitArgMain
      (
       argc,argv,
       //mandatory arguments
       LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile),
       //optional arguments
       LArgMain()  << EAM(aInHomolDirName, "HomolIn", true, "Input Homol directory suffix (without \"Homol\")")
                   << EAM(aNumWindows, "NbWin", true, "Minimal homol points in each picture (default: 100)")
                   << EAM(aOutHomolDirName, "HomolOut", true, "Output Homol directory suffix (default: _mini)")
                   << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
      );

    if (MMVisualMode) return EXIT_SUCCESS;

    std::cout<<"Minimal number of points per picture: "<<aNumWindows<<std::endl;

    // Initialize name manipulator & files
    std::string aDirXML,aDirImages,aPatIm;
    //std::string aGCPFileTmpName;
    //SplitDirAndFile(aDirXML,aGCPFileTmpName,aGCPFileName);
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;


    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


    //create pictures list, and pictures size list ---------------------
    std::vector<cPic*> allPics;
    std::vector<cPicSize> allPicSizes;

    std::cout<<"Found pictures:\n";
    for (unsigned int i=0;i<aSetIm.size();i++)
    {
        std::cout<<" - "<<aSetIm[i]<<": ";
        Tiff_Im aPic(aSetIm[i].c_str());
        Pt2di aPicSize=aPic.sz();
        std::cout<<aPicSize<<"\n";
        allPics.push_back(new cPic(aSetIm[i],allPicSizes,aNumWindows));
    }
    std::cout<<"All sizes: \n";
    for (unsigned int i=0;i<allPicSizes.size();i++)
    {
        std::cout<<"  * "<<allPicSizes[i].getPicSz()<<" => "<<allPicSizes[i].getNbWin()<<" windows of "<<allPicSizes[i].getWinSz()<<" pixels\n";
    }
    
    //read all homol points --------------------------------------------
    
    // Init Keys for homol files
    std::list<cHomol*> allHomolsIn;
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aInHomolDirName)
            +  std::string("@")
            +  std::string(anExt);

    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aOutHomolDirName)
            +  std::string("@")
            +  std::string(anExt);
    
    std::cout<<"Read Homol points:\n";
    for (unsigned int i=0;i<allPics.size();i++)
    {
        cPic *pic1=allPics[i];
        std::cout<<" Picture "<<aSetIm[i]<<": \n";
        for (unsigned int j=i+1;j<allPics.size();j++)
        {
            cPic *pic2=allPics[j];
            std::string aNameIn = aDirImages + aICNM->Assoc1To2(aKHIn,pic1->getName(),pic2->getName(),true);
            if (ELISE_fp::exist_file(aNameIn))
            {
                ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
                cout<<aNameIn<<"  Pack size: "<<aPackIn.size()<<"\n";
                for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; ++itP)
                {
                    Pt2dr aP1 = itP->P1();
                    Pt2dr aP2 = itP->P2();
                    #ifdef ReductHomolImage_DEBUG
                    std::cout<<"For pair : "<<aP1<<" "<<aP2<<std::endl;
                    #endif
                    
                    //search if already exists :
                    cPointOnPic *aPointOnPic1=0;
                    cPointOnPic *aPointOnPic2=0;
                    std::list<cPointOnPic*>::iterator itPointOnPic;
                    for (itPointOnPic=pic1->getAllPointsOnPic()->begin();
                         itPointOnPic!=pic1->getAllPointsOnPic()->end();
                         ++itPointOnPic)
                    {
                        Pt2dr aPtH=(*itPointOnPic)->getPt();
                        if ((fabs(aPtH.x-aP1.x)<0.1)&&
                            (fabs(aPtH.y-aP1.y)<0.1))
                        {
                            aPointOnPic1=(*itPointOnPic);
                            break;
                        }
                    }
                    for (itPointOnPic=pic2->getAllPointsOnPic()->begin();
                         itPointOnPic!=pic2->getAllPointsOnPic()->end();
                         ++itPointOnPic)
                    {
                        Pt2dr aPtH=(*itPointOnPic)->getPt();
                        if ((fabs(aPtH.x-aP2.x)<0.1)&&
                            (fabs(aPtH.y-aP2.y)<0.1))
                        {
                            aPointOnPic2=(*itPointOnPic);
                            break;
                        }
                    }
                    
                    #ifdef ReductHomolImage_DEBUG
                    std::cout<<"Result Homol: "<<aPointOnPic1<<" "<<aPointOnPic2<<std::endl;
                    #endif
                    
                    if (aPointOnPic1 && (!aPointOnPic2))
                        aPointOnPic1->getHomol()->add(pic2,aP2);
                    else if (aPointOnPic2 && (!aPointOnPic1))
                        aPointOnPic2->getHomol()->add(pic1,aP1);
                    else if (aPointOnPic1 && aPointOnPic2 &&(aPointOnPic1->getHomol()!=aPointOnPic2->getHomol()))
                    {
                        cPointOnPic * aPointOnPic12=aPointOnPic1->getHomol()->getPointOnPic(pic2);
                        cPointOnPic * aPointOnPic21=aPointOnPic2->getHomol()->getPointOnPic(pic1);
                        if (
                            ((aPointOnPic21)
                              &&(
                                 (fabs(aPointOnPic21->getPt().x-aP2.x)>0.2)
                                 ||(fabs(aPointOnPic21->getPt().y-aP2.y)>0.2))
                                )
                            ||
                            ((aPointOnPic12)
                              &&(
                                 (fabs(aPointOnPic12->getPt().x-aP1.x)>0.2)
                                 ||(fabs(aPointOnPic12->getPt().y-aP1.y)>0.2))
                                )
                           )
                        {
                            std::cout<<"Bad homols!\n";
                            aPointOnPic1->getHomol()->setBad();
                            aPointOnPic2->getHomol()->setBad();
                        }
                        
                        //merge the two homol points
                        /*homolP1->add(homolP2);
                        homolP2->getPointOnPics().clear();
                        allHomolsIn.erase(itHomolP2);*/
                    }
                    else if ((!aPointOnPic1) && (!aPointOnPic2))
                    {
                        //new homol point
                        allHomolsIn.push_back(new cHomol);
                        allHomolsIn.back()->add(pic1,aP1);
                        allHomolsIn.back()->add(pic2,aP2);
                    }
                }
            }
        }
    }
    #ifdef ReductHomolImage_DEBUG
    std::cout<<"Found "<<allHomolsIn.size()<<" Homol points :\n";
    std::list<cHomol*>::iterator itHomol;
    for (itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
    {
        //std::cout<<(*itHomol)->getPointOnPics().size()<<" ";
        (*itHomol)->print();
    }
    std::cout<<std::endl;
    #endif
    
    //sort pics on number of homols ------------------------------------
    std::sort(allPics.begin(), allPics.end(), compareNumberOfHomolPics);
    
    #ifdef ReductHomolImage_DEBUG
    std::cout<<"Homols per image:";
    for (unsigned int i=0;i<allPics.size();i++)
    {
        allPics[i]->printHomols();
        /*std::cout<<std::endl<<"  - "<<allPics[i]->getName()<<" "<<std::flush;
        std::list<cPointOnPic*>::iterator itHomolPoint;
        for (itHomolPoint=allPics[i]->getAllPointsOnPic()->begin();
             itHomolPoint!=allPics[i]->getAllPointsOnPic()->end();
             ++itHomolPoint)
        {
            std::cout<<(*itHomolPoint)->getHomol()<<" "<<std::flush;
            std::cout<<(*itHomolPoint)->getHomol()->getId()<<" "<<std::flush;
        }*/
    }
    std::cout<<std::endl;
    #endif
    
    
    //create new homols ------------------------------------------------
    std::cout<<"Create new homol"<<std::endl;
    
    for (unsigned int i=0;i<allPics.size();i++)
    {
        cPic* aPic=allPics[i];
        std::cout<<"For "<<aPic->getName()<<"\n";
        //for each window, get current best selected point, and best existing point
        for (int x=0;x<aPic->getPicSize()->getNbWin().x;x++)
        {
            for (int y=0;y<aPic->getPicSize()->getNbWin().y;y++)
            {
                //get window
                int aXmin=x*aPic->getPicSize()->getWinSz().x;
                int aXmax=(x+1)*aPic->getPicSize()->getWinSz().x;
                int aYmin=y*aPic->getPicSize()->getWinSz().y;
                int aYmax=(y+1)*aPic->getPicSize()->getWinSz().y;
                #ifdef ReductHomolImage_DEBUG
                std::cout<<"    For window ("<<aXmin<<","<<aYmin<<") ("<<aXmax<<","<<aYmax<<"): ";
                #endif
                
                //best old Homol in this window
                cPointOnPic *aBestOldPointOnPic=0;
                unsigned int aBestOldHomolMultiple=0;
                std::list<cPointOnPic*>::iterator itHomolPoint;
                for (itHomolPoint=aPic->getAllPointsOnPic()->begin();
                     itHomolPoint!=aPic->getAllPointsOnPic()->end();
                     ++itHomolPoint)
                {
                    if ((*itHomolPoint)->getHomol()->isBad()) continue;
                    
                    if (((*itHomolPoint)->getPt().x>aXmin)&&((*itHomolPoint)->getPt().x<aXmax)&&
                        ((*itHomolPoint)->getPt().y>aYmin)&&((*itHomolPoint)->getPt().y<aYmax))
                    {
                        if ((!aBestOldPointOnPic)
                            ||((*itHomolPoint)->getHomol()->getPointOnPics().size()>aBestOldHomolMultiple ))
                        {
                            aBestOldPointOnPic=(*itHomolPoint);
                            aBestOldHomolMultiple=aBestOldPointOnPic->getHomol()->getPointOnPics().size();
                        }
                    }
                }
                //best selected homol in this window
                cPointOnPic *aBestSelectedPointOnPic=0;
                unsigned int aBestSelectedHomolMultiple=0;
                for (itHomolPoint=aPic->getAllSelectedPointsOnPic()->begin();
                     itHomolPoint!=aPic->getAllSelectedPointsOnPic()->end();
                     ++itHomolPoint)
                {
                    if (((*itHomolPoint)->getPt().x>aXmin)&&((*itHomolPoint)->getPt().x<aXmax)&&
                        ((*itHomolPoint)->getPt().y>aYmin)&&((*itHomolPoint)->getPt().y<aYmax))
                    {
                        if ((!aBestSelectedPointOnPic)
                            ||((*itHomolPoint)->getHomol()->getPointOnPics().size()>aBestSelectedHomolMultiple ))
                        {
                            aBestSelectedPointOnPic=(*itHomolPoint);
                            aBestSelectedHomolMultiple=aBestSelectedPointOnPic->getHomol()->getPointOnPics().size();
                        }
                    }
                }
                

                //if best existing is better than best selected, add best existing
                #ifdef ReductHomolImage_DEBUG
                std::cout<<"    BestOldHomolMultiple: "<<aBestOldHomolMultiple;
                std::cout<<"    BestSelectedHomolMultiple: "<<aBestSelectedHomolMultiple;
                //if (aBestOldHomolMultiple>0)
                //    //std::cout<<"    BestOldHomol: "<<aBestOldPointOnPic->getHomol()->getId();
                //    std::cout<<"    BestOldHomol: "<<aBestOldPointOnPic->getPt();
                //if (aBestSelectedHomolMultiple>0)
                //    //std::cout<<"    BestSelectedHomol: "<<aBestSelectedPointOnPic->getHomol()->getId();
                //    std::cout<<"    BestSelectedHomol: "<<aBestSelectedPointOnPic->getPt();
                #endif
                if (aBestOldHomolMultiple>aBestSelectedHomolMultiple)
                {
                    //add this homol to every picture it is in!
                    std::vector<cPointOnPic*>::iterator itPointOnPic;
                    cHomol * aHomol=aBestOldPointOnPic->getHomol();
                    
                    for (itPointOnPic=aHomol->getPointOnPics().begin();
                         itPointOnPic!=aHomol->getPointOnPics().end();
                         ++itPointOnPic)
                    {
                        cPic * aOtherPic=(*itPointOnPic)->getPic();
                        aOtherPic->addSelectedPointOnPicUnique((*itPointOnPic));
                    }
                    
                }
                #ifdef ReductHomolImage_DEBUG
                if (aBestOldHomolMultiple==0){
                    std::cout<<"      Warning: "<<aPic->getName()<<" lacks Homol points!";
                }
                std::cout<<std::endl;
                #endif
                if (aBestOldHomolMultiple>0)
                    aPic->incNbWinUsed();
            }
        }
    }

    #ifdef ReductHomolImage_DEBUG
    std::cout<<"New Homols per image:";
    for (unsigned int i=0;i<allPics.size();i++)
    {
        std::cout<<std::endl<<"  - "<<allPics[i]->getName()<<" "<<std::flush;
        std::list<cPointOnPic*>::iterator itPointsOnPic;
        for (itPointsOnPic=allPics[i]->getAllSelectedPointsOnPic()->begin();
             itPointsOnPic!=allPics[i]->getAllSelectedPointsOnPic()->end();
             ++itPointsOnPic)
        {
            std::cout<<(*itPointsOnPic)->getHomol()->getId()<<" "<<std::flush;
        }
    }
    std::cout<<std::endl;
    #endif
    
    std::cout<<"Write new Packs:\n";
    std::ofstream aFileBadPictureNames;
    aFileBadPictureNames.open("Schnaps_pictures_poubelle.txt");
    if (!aFileBadPictureNames.is_open())
    {
        std::cout<<"Impossible to create \"Schnaps_pictures_poubelle.txt\" file!\n";
        return -1;
    }
    for (unsigned int i=0;i<allPics.size();i++)
    {
        cPic *pic1=allPics[i];
        std::cout<<" - "<<pic1->getName()<<": "<<pic1->getPercentWinUsed()<<"% of the picture covered ("<<pic1->getAllSelectedPointsOnPic()->size()<<" points)\n";
        if (pic1->getPercentWinUsed()<25)
            aFileBadPictureNames<<pic1->getName()<<"\n";
        for (unsigned int j=i+1;j<allPics.size();j++)
        {
            cPic *pic2=allPics[j];
            
            std::string aNameOut1 = aDirImages + aICNM->Assoc1To2(aKHOut,pic1->getName(),pic2->getName(),true);
            std::string aNameOut2 = aDirImages + aICNM->Assoc1To2(aKHOut,pic1->getName(),pic2->getName(),true);
            ElPackHomologue aPackOut1;
            ElPackHomologue aPackOut2;
            std::list<cPointOnPic*>::iterator itPointsOnPic;
            for (itPointsOnPic=pic1->getAllSelectedPointsOnPic()->begin();
                 itPointsOnPic!=pic1->getAllSelectedPointsOnPic()->end();
                 ++itPointsOnPic)
            {
                cPointOnPic* aPointOnPic2=(*itPointsOnPic)->getHomol()->getPointOnPic(pic2);
                if (!aPointOnPic2) continue;
                Pt2dr aP1=(*itPointsOnPic)->getPt();
                Pt2dr aP2=aPointOnPic2->getPt();
                ElCplePtsHomologues aCple1(aP1,aP2);
                aPackOut1.Cple_Add(aCple1);
                ElCplePtsHomologues aCple2(aP1,aP2);
                aPackOut2.Cple_Add(aCple2);
            }
            
            if (aPackOut1.size()>0)
            {
                aPackOut1.StdPutInFile(aNameOut1);
                aPackOut2.StdPutInFile(aNameOut2);
            }
        }
    }
    aFileBadPictureNames.close();

  /*todo :
   * error if no packs
   * warning if too few points on images
   * print stats
   * merge homol points?
   * */
  
  
  std::cout<<"Quit"<<std::endl;

  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
