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
#include <iterator>
#include <map>

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
//#define ReductHomolImage_VeryStrict_DEBUG
#define ReductHomolImage_UsefullPackSize 100
#define ReductHomolImage_UselessPackSize 50

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
    int getUsageBuffer(){return mUsageBuffer;}

  protected:
    Pt2di mPicSz;
    Pt2di mWinSz;
    Pt2di mNbWin;
    int mUsageBuffer;//buffer for image usage check
  };


cPicSize::cPicSize(Pt2di aSz,int aNumWindows) :
    mPicSz(aSz)
{
    float aXYratio=((float)aSz.x)/aSz.y;
    mNbWin.x=sqrt(aNumWindows)*sqrt(aXYratio)+1;
    mNbWin.y=sqrt(aNumWindows)/sqrt(aXYratio)+1;
    mWinSz.x=((float)mPicSz.x)/mNbWin.x+0.5;
    mWinSz.y=((float)mPicSz.y)/mNbWin.y+0.5;
    mUsageBuffer=mNbWin.x/20;//where the arbitrary buffer size is calculated
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
    int getId(){return mId;}
    cPic* getPic(){return mPic;}
    Pt2dr& getPt(){return mPt;}
    cHomol* getHomol(){return mHomol;}
    void setHomol(cHomol* aHomol){mHomol=aHomol;}
    void print();
  protected:
    int mId;//unique id
    static int mHomolOnPicCounter;
    cPic* mPic;
    Pt2dr mPt;
    cHomol* mHomol;//the Homol it cames from
    
};
int cPointOnPic::mHomolOnPicCounter=0;

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
    bool add(cPic * aPic, Pt2dr aPt);//return true if not already in
    void add(cHomol *aHomol);
    void print();
    bool isBad(){return mBad;}
    void setBad(){mBad=true;}
    cPointOnPic* getPointOnPic(cPic * aPic);
    void addAppearsOnCouple(cPic * aPicA,cPic * aPicB);
    bool appearsOnCouple(cPic * aPicA,cPic * aPicB);
    bool appearsOnCouple2way(cPic * aPicA,cPic * aPicB);
    cPic* getAppearsOnCoupleA(unsigned int i){return mAppearsOnCoupleA[i];}
    cPic* getAppearsOnCoupleB(unsigned int i){return mAppearsOnCoupleB[i];}
    int getAppearsOnCoupleSize(){return mAppearsOnCoupleA.size();}
  protected:
    int mId;//unique id
    static int mHomolCounter;
    std::vector<cPointOnPic*> mPointOnPics;
    bool mBad;
    std::vector<cPic*> mAppearsOnCoupleA;//record in which couple A-B it has been seen
    std::vector<cPic*> mAppearsOnCoupleB;
};
int cHomol::mHomolCounter=0;


//----------------------------------------------------------------------------
// Pic class: a picture with its homol points
class cPic
{
  public:
    cPic(std::string aDir,std::string aName,std::vector<cPicSize> & allSizes,int aNumWindows);
    cPic(cPic* aPic);
    std::string getName(){return mName;}
    cPicSize * getPicSize(){return mPicSize;}
    bool removeHomolPoint(cPointOnPic* aPointOnPic);
    void printHomols();
    bool addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic);
    float getPercentWinUsed();
    //void incNbWinUsed(){mNbWinUsed++;}
    void setWinUsed(int _x,int _y);
    cPointOnPic * findPointOnPic(Pt2dr & aPt);
    void addPointOnPic(cPointOnPic* aPointOnPic);
    int getAllPointsOnPicSize(){return mAllPointsOnPic.size();}
    int getAllSelectedPointsOnPicSize(){return mAllSelectedPointsOnPic.size();}
    void selectHomols();
    void fillPackHomol(cPic* aPic2,std::string& aNameOut1,std::string& aNameOut2);
  protected:
    std::string mName;
    cPicSize * mPicSize;
    std::map<double,cPointOnPic*> mAllPointsOnPic;//key: x+mPicSize->getPicSz().x*y
    std::map<double,cPointOnPic*> mAllSelectedPointsOnPic;//key: x+mPicSize->getPicSz().x*y
    std::vector<bool> mWinUsed;//to check repartition of homol points (with buffer)
    //int mNbWinUsed;
    double makePOPKey(Pt2dr & aPt){return aPt.x*100+mPicSize->getPicSz().x*100*aPt.y*100;}
};

//----------------------------------------------------------------------------

class CompiledKey2
{
  public:
    CompiledKey2(cInterfChantierNameManipulateur * aICNM,std::string aKH);
    std::string get(std::string param1,std::string param2);
    std::string getDir(std::string param1,std::string param2);
    std::string getFile(std::string param1,std::string param2);
    std::string getSuffix(){return mPart3;}
  protected:
    cInterfChantierNameManipulateur * mICNM;
    std::string mKH,mPart1,mPart2,mPart3;
};

//----------------------------------------------------------------------------

cPointOnPic::cPointOnPic(cPic *aPic,Pt2dr aPt,cHomol* aHomol) :
    mPic(aPic),mPt(aPt),mHomol(aHomol)
{
    mId=mHomolOnPicCounter;
    mHomolOnPicCounter++;

#ifdef ReductHomolImage_DEBUG
    cout<<"New cPointOnPic "<<this<<" for Homol "<<mHomol->getId()<<"   homol "<<mHomol<<endl;
#endif
}

void cPointOnPic::print()
{
    cout<<"     cPointOnPic "<<mId<<": "<<this<<" on "<<mPic->getName()<<": "<<mPt<<"   homol "<<mHomol<<endl;
}

void cPic::addPointOnPic(cPointOnPic* aPointOnPic)
{
    mAllPointsOnPic.insert(
        std::make_pair<double,cPointOnPic*>(
            makePOPKey(aPointOnPic->getPt()),
            aPointOnPic));
}

cPointOnPic * cPic::findPointOnPic(Pt2dr & aPt)
{
    std::map<double,cPointOnPic*>::iterator it;
    it=mAllPointsOnPic.find(makePOPKey(aPt));
    return (it==mAllPointsOnPic.end()) ? 0 : it->second;
    /*if (it==mAllPointsOnPic.end())
        return 0;
    else
        return (it->second->getHomol()->isBad() ? 0 : it->second);*/
    
    /*std::list<cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=getAllPointsOnPic()->begin();
         itPointOnPic!=getAllPointsOnPic()->end();
         ++itPointOnPic)
    {
        //if ((*itPointOnPic)->getHomol()->isBad()) continue;
        const Pt2dr & aPtH=(*itPointOnPic)->getPt();
        //if ((fabs(aPtH.x-aP1.x)<0.01)&&(fabs(aPtH.y-aP1.y)<0.01))
        if ((aPtH.x==aPt.x)&&(aPtH.y==aPt.y))
        {
            return (*itPointOnPic);
        }
    }
    return 0;*/
}


cPointOnPic* cHomol::getPointOnPic(cPic * aPic)
{
    for (unsigned int i=0;i<mPointOnPics.size();i++)
        if (mPointOnPics[i]->getPic()==aPic)
            return mPointOnPics[i];
    return 0;
}
//----------------------------------------------------------------------------

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
    for (unsigned int i=0;i<mAppearsOnCoupleA.size();i++)
    {
        cout<<"   seen on "<<mAppearsOnCoupleA[i]->getName()<<" "<<mAppearsOnCoupleB[i]->getName()<<"\n";
    }
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

bool cHomol::add(cPic * aPic, Pt2dr aPt)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"add PointOnPic "<<aPic->getName()<<" "<<aPt<<" on Homol "<<mId<<" \n";
#endif
    //test if this homol already has this pic
    std::vector<cPointOnPic*>::iterator itHomolPoint;
    for (itHomolPoint=mPointOnPics.begin();itHomolPoint!=mPointOnPics.end();++itHomolPoint)
    {
        if ((*itHomolPoint)->getPic()==aPic)
        {
            //std::cout<<"Bad Homol!\n";
            setBad();
            return false;
        }
    }

    mPointOnPics.push_back(new cPointOnPic(aPic,aPt,this));
    aPic->addPointOnPic(mPointOnPics.back());
    return true;
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
        aHomol->getPointOnPics()[i]->getPic()->addPointOnPic(mPointOnPics.back());
    }
    for (unsigned int i=0;i<aHomol->mAppearsOnCoupleA.size();i++)
    {
        mAppearsOnCoupleA.push_back(aHomol->mAppearsOnCoupleA[i]);
        mAppearsOnCoupleB.push_back(aHomol->mAppearsOnCoupleB[i]);
    }
}


void cHomol::addAppearsOnCouple(cPic * aPicA,cPic * aPicB)
{
    mAppearsOnCoupleA.push_back(aPicA);
    mAppearsOnCoupleB.push_back(aPicB);
}

bool cHomol::appearsOnCouple(cPic * aPicA,cPic * aPicB)
{
    for (unsigned int i=0;i<mAppearsOnCoupleA.size();i++)
    {
        if ((mAppearsOnCoupleA[i]==aPicA)&&(mAppearsOnCoupleB[i]==aPicB))
            return true;
    }
    return false;
}

bool cHomol::appearsOnCouple2way(cPic * aPicA,cPic * aPicB)
{
    bool seen_way1=false;
    bool seen_way2=false;
    for (unsigned int i=0;i<mAppearsOnCoupleA.size();i++)
    {
        //must be seen on both directions
        if ((mAppearsOnCoupleA[i]==aPicA)&&(mAppearsOnCoupleB[i]==aPicB))
        {
            seen_way1=true;
            if (seen_way2) return true;
        }
        if ((mAppearsOnCoupleA[i]==aPicB)&&(mAppearsOnCoupleB[i]==aPicA))
        {
            seen_way2=true;
            if (seen_way1) return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------------

cPic::cPic(std::string aDir,std::string aName,std::vector<cPicSize> & allSizes,int aNumWindows) :
    mName(aName),mPicSize(0)//,mNbWinUsed(0)
{
    Tiff_Im aPic( Tiff_Im::StdConvGen(aDir+"/"+aName,1,false)); //to read file in Tmp-MM-Dir if needed
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
    mWinUsed.resize(mPicSize->getNbWin().x*mPicSize->getNbWin().y,false);
}

cPic::cPic(cPic * aPic) :
    mName(aPic->mName),mPicSize(aPic->mPicSize)
{
}

void cPic::printHomols()
{
    std::cout<<mName<<" homols:\n";
    std::map<double,cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=mAllPointsOnPic.begin();itPointOnPic!=mAllPointsOnPic.end();++itPointOnPic)
    {
        std::cout<<(itPointOnPic->second)<<" "<<(itPointOnPic->second)->getPt()<<" "<<(itPointOnPic->second)->getHomol()
                 <<" multi "<<(itPointOnPic->second)->getHomol()->getPointOnPics().size()<<std::endl;
    }
}

bool cPic::addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic)
{
    std::map<double,cPointOnPic*>::iterator it;
    it=mAllSelectedPointsOnPic.find(makePOPKey(aPointOnPic->getPt()));
    if (it!=mAllSelectedPointsOnPic.end())
        return false;
    else
        mAllSelectedPointsOnPic.insert(
            std::make_pair<double,cPointOnPic*>(
                makePOPKey(aPointOnPic->getPt()),
                aPointOnPic));
    return true;
}

void cPic::setWinUsed(int _x,int _y)
{
    int bufSz=getPicSize()->getUsageBuffer();
    for (int x=_x-bufSz;x<_x+bufSz+1;x++)
    {
        if ((x<0)||(x>=mPicSize->getNbWin().x)) continue;
        for (int y=_y-bufSz;y<_y+bufSz+1;y++)
        {
            if ((y<0)||(y>=mPicSize->getNbWin().y)) continue;
            mWinUsed[x+y*mPicSize->getNbWin().x]=true;
        }
    }
    
}

float cPic::getPercentWinUsed()
{
    //return 100.0*((float)mNbWinUsed)/(mPicSize->getNbWin().x*mPicSize->getNbWin().y);
    int nbWinUsed=0;
    for (int i=0;i<mPicSize->getNbWin().x*mPicSize->getNbWin().y;i++)
        if (mWinUsed[i]) nbWinUsed++;
    return 100.0*((float)nbWinUsed)/(mPicSize->getNbWin().x*mPicSize->getNbWin().y);
}


bool cPic::removeHomolPoint(cPointOnPic* aPointOnPic)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"cPic::removeHomolPoint"<<std::endl;
#endif

    return (mAllPointsOnPic.erase(makePOPKey(aPointOnPic->getPt())) == 1);
    
}



void cPic::selectHomols()
{
    std::vector< std::vector<cPointOnPic*> > winBestPoP;
    std::vector< std::vector<unsigned int> > winBestMulti;
    winBestPoP.resize(mPicSize->getNbWin().y);
    for (unsigned int i=0;i<winBestPoP.size();i++)
        winBestPoP[i].resize(mPicSize->getNbWin().x,0);
    winBestMulti.resize(mPicSize->getNbWin().y);
    for (unsigned int i=0;i<winBestMulti.size();i++)
        winBestMulti[i].resize(mPicSize->getNbWin().x,0);
    
    std::map<double,cPointOnPic*>::iterator itHomolPoint;
    int x,y;
    //compute best already-selected homol
    for (itHomolPoint=mAllSelectedPointsOnPic.begin();
         itHomolPoint!=mAllSelectedPointsOnPic.end();
         ++itHomolPoint)
    {
        cPointOnPic* aPoP=(*itHomolPoint).second;
        if (aPoP->getHomol()->isBad()) continue;
        x=aPoP->getPt().x/getPicSize()->getWinSz().x;
        y=aPoP->getPt().y/getPicSize()->getWinSz().y;
        
        if (x>=getPicSize()->getNbWin().x) x=getPicSize()->getNbWin().x-1;
        if (y>=getPicSize()->getNbWin().y) y=getPicSize()->getNbWin().y-1;
        //cout<<"already "<<x<<" "<<y<<" "<<aPoP->getHomol()->getPointOnPics().size()<<endl;
        if (winBestMulti[y][x]<aPoP->getHomol()->getPointOnPics().size())
        {
            winBestMulti[y][x]=aPoP->getHomol()->getPointOnPics().size();
            //we update winBestMulti but not winBestPoP, because we won't add this point again
        }
    }
    
    //search if exists better homol
    for (itHomolPoint=mAllPointsOnPic.begin();
         itHomolPoint!=mAllPointsOnPic.end();
         ++itHomolPoint)
    {
        cPointOnPic* aPoP=(*itHomolPoint).second;
        if (aPoP->getHomol()->isBad()) continue;
        x=aPoP->getPt().x/getPicSize()->getWinSz().x;
        y=aPoP->getPt().y/getPicSize()->getWinSz().y;
        if (x>=getPicSize()->getNbWin().x) x=getPicSize()->getNbWin().x-1;
        if (y>=getPicSize()->getNbWin().y) y=getPicSize()->getNbWin().y-1;
        //cout<<"old "<<x<<" "<<y<<" "<<aPoP->getHomol()->getPointOnPics().size()<<endl;

        if (winBestMulti[y][x]<aPoP->getHomol()->getPointOnPics().size())
        {
            winBestMulti[y][x]=aPoP->getHomol()->getPointOnPics().size();
            winBestPoP[y][x]=aPoP;
        }
    }
    
    //for each window, add best point if needed
    for (x=0;x<getPicSize()->getNbWin().x;x++)
    {
        for (y=0;y<getPicSize()->getNbWin().y;y++)
        {
            cPointOnPic *aBestSelectedPointOnPic=winBestPoP[y][x];
            if (aBestSelectedPointOnPic)
            {
                //add this homol to every picture it is in!
                std::vector<cPointOnPic*>::iterator itPointOnPic;
                cHomol * aHomol=aBestSelectedPointOnPic->getHomol();
                
                for (itPointOnPic=aHomol->getPointOnPics().begin();
                     itPointOnPic!=aHomol->getPointOnPics().end();
                     ++itPointOnPic)
                {
                    cPic * aOtherPic=(*itPointOnPic)->getPic();
                    aOtherPic->addSelectedPointOnPicUnique((*itPointOnPic));
                }
            }
            if (winBestMulti[y][x]>0)
                setWinUsed(x,y);
        }
    }
}


void cPic::fillPackHomol(cPic* aPic2,std::string& aNameOut1,std::string& aNameOut2)
{
    ElPackHomologue aPackOut1;
    ElPackHomologue aPackOut2;
    std::map<double,cPointOnPic*>::iterator itPointsOnPic;
    for (itPointsOnPic=mAllSelectedPointsOnPic.begin();
         itPointsOnPic!=mAllSelectedPointsOnPic.end();
         ++itPointsOnPic)
    {
        cPointOnPic* aPointOnPic1=(*itPointsOnPic).second;
        cPointOnPic* aPointOnPic2=aPointOnPic1->getHomol()->getPointOnPic(aPic2);
        if (!aPointOnPic2) continue;
        Pt2dr aP1=aPointOnPic1->getPt();
        Pt2dr aP2=aPointOnPic2->getPt();
        ElCplePtsHomologues aCple1(aP1,aP2);
        aPackOut1.Cple_Add(aCple1);
        ElCplePtsHomologues aCple2(aP2,aP1);
        aPackOut2.Cple_Add(aCple2);
    }
    
    if (aPackOut1.size()>0)
    {
        //std::cout<<aNameOut1<<": "<<aPackOut1.size()<<" pairs."<<endl;
        //std::cout<<aNameOut2<<": "<<aPackOut2.size()<<" pairs."<<endl;
        aPackOut1.StdPutInFile(aNameOut1);
        aPackOut2.StdPutInFile(aNameOut2);
    }
    
}

bool compareNumberOfHomolPics (cPic* aPic1,cPic* aPic2)
{
    return (aPic1->getAllPointsOnPicSize()<aPic2->getAllPointsOnPicSize());
}


//----------------------------------------------------------------------------

CompiledKey2::CompiledKey2(cInterfChantierNameManipulateur * aICNM,std::string aKH):
    mICNM(aICNM),mKH(aKH)
{
    std::string aNameIn = mICNM->Assoc1To2(aKH,"$1","$2",true);
    std::size_t pos1 = aNameIn.find("$1");
    std::size_t pos2 = aNameIn.find("$2");
    mPart1 = aNameIn.substr (0,pos1);
    mPart2 = aNameIn.substr (pos1+2,pos2-pos1-2);
    mPart3 = aNameIn.substr (pos2+2);
}

std::string CompiledKey2::get(std::string param1,std::string param2)
{
    return mPart1+param1+mPart2+param2+mPart3;
}

std::string CompiledKey2::getDir(std::string param1,std::string param2)
{
    std::string aStr=mPart1+param1+mPart2+param2+mPart3;
    std::size_t pos1 = aStr.rfind("/");
    return aStr.substr(0,pos1+1);
}

std::string CompiledKey2::getFile(std::string param1,std::string param2)
{
    std::string aStr=mPart1+param1+mPart2+param2+mPart3;
    std::size_t pos1 = aStr.rfind("/");
    return aStr.substr(pos1+1);   
}



int schnaps_main(int argc,char ** argv)
{
    std::string aFullPattern;//pattern of all images
    std::string aInHomolDirName="";//input Homol dir suffix
    std::string aOutHomolDirName="_mini";//output Homol dir suffix
    std::string aPoubelleName="Schnaps_poubelle.txt";
    int aNumWindows=1000;//minimal homol points in each picture
    bool ExpTxt=false;//Homol are in dat or txt
    bool veryStrict=false;

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
                   << EAM(aNumWindows, "NbWin", true, "Minimal homol points in each picture (default: 1000)")
                   << EAM(aOutHomolDirName, "HomolOut", true, "Output Homol directory suffix (default: _mini)")
                   << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                   << EAM(veryStrict,"VeryStrict",true,"Be very strict with homols (remove any suspect), def=false")
                   << EAM(aPoubelleName,"PoubelleName",true,string("Where to write suspicious pictures names, def=\"")+aPoubelleName+"\"")
      );

    if (MMVisualMode) return EXIT_SUCCESS;

    std::cout<<"Number of searching windows: "<<aNumWindows<<std::endl;

    // Initialize name manipulator & files
    std::string aDirXML,aDirImages,aPatIm;
    //std::string aGCPFileTmpName;
    //SplitDirAndFile(aDirXML,aGCPFileTmpName,aGCPFileName);
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;


    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


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
    
    CompiledKey2 aCKin(aICNM,aKHIn);
    CompiledKey2 aCKout(aICNM,aKHOut);

    //create pictures list, and pictures size list ---------------------
    //std::vector<cPic*> allPics;
    std::map<std::string,cPic*> allPics;
    
    std::vector<cPicSize> allPicSizes;

    std::cout<<"Found "<<aSetIm.size()<<" pictures."<<endl;
    for (unsigned int i=0;i<aSetIm.size();i++)
    {
        //std::cout<<" - "<<aSetIm[i]<<"\n";
        //Tiff_Im aPic(aSetIm[i].c_str());
        /*Tiff_Im aPic(Tiff_Im::StdConvGen(aDirImages+"/"+aSetIm[i],1,false)   );
        Pt2di aPicSize=aPic.sz();
        std::cout<<aPicSize<<"\n";*/
        //allPics.push_back(new cPic(aDirImages,aSetIm[i],allPicSizes,aNumWindows));
        allPics.insert(std::make_pair<std::string,cPic*>(aSetIm[i]+aCKin.getSuffix(),new cPic(aDirImages,aSetIm[i],allPicSizes,aNumWindows)));
    }

    ELISE_ASSERT(aSetIm.size()>0,"ERROR: No image found!");

    std::cout<<"All sizes: \n";
    for (unsigned int i=0;i<allPicSizes.size();i++)
    {
        std::cout<<"  * "<<allPicSizes[i].getPicSz()<<" => "<<allPicSizes[i].getNbWin()<<" windows of "<<allPicSizes[i].getWinSz()<<" pixels"<<endl;
    }
    
    
    //read all homol points --------------------------------------------
    
    
    std::cout<<"Read Homol points:"<<endl;
    
    std::map<std::string,cPic*>::iterator itPic1,itPic2;
    int i=0;
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    //for (unsigned int i=0;i<allPics.size();i++)
    {
        cPic *pic1=(*itPic1).second;//allPics[i];
        std::cout<<" Picture "<<pic1->getName()<<": ";
        //get all pictures having pac with pic1
        cInterfChantierNameManipulateur * homolICNM=cInterfChantierNameManipulateur::BasicAlloc(aCKin.getDir(pic1->getName(),aPatIm));
        std::list<std::string> aSetPac = homolICNM->StdGetListOfFile(aCKin.getFile(pic1->getName(),aPatIm));
                
        std::cout<<"Found "<<aSetPac.size()<<" pacs."<<endl;
        
        //if (i==20) return 0;
        i++;
        std::list<std::string>::iterator itPacName;
        for (itPacName=aSetPac.begin();itPacName!=aSetPac.end();++itPacName)
        {
            cPic *pic2=allPics.at( (*itPacName) );
            
            std::string aNameIn1= aDirImages + aCKin.get(pic1->getName(),pic2->getName());
            //if (ELISE_fp::exist_file(aNameIn1))
            {
                ElPackHomologue aPackIn1 =  ElPackHomologue::FromFile(aNameIn1);
                //cout<<aNameIn1<<"  Pack size: "<<aPackIn1.size()<<"\n";
                for (ElPackHomologue::const_iterator itP=aPackIn1.begin(); itP!=aPackIn1.end() ; ++itP)
                {
                    Pt2dr aP1 = itP->P1();
                    Pt2dr aP2 = itP->P2();


                    #ifdef ReductHomolImage_DEBUG
                    std::cout<<" On "<<aNameIn1<<" for pair : "<<aP1<<" "<<aP2<<" => ";
                    #endif

                    //search if already exists :
                    cPointOnPic *aPointOnPic1=pic1->findPointOnPic(aP1);
                    cPointOnPic *aPointOnPic2=pic2->findPointOnPic(aP2);
                    
                    
                    #ifdef ReductHomolImage_DEBUG
                    std::cout<<" Result Homol: "<<aPointOnPic1<<" "<<aPointOnPic2<<std::endl;
                    if (aPointOnPic1) aPointOnPic1->getHomol()->print();
                    if (aPointOnPic2) aPointOnPic2->getHomol()->print();
                    #endif
                    
                    /*if (((fabs(aP1.x-494.410)<0.1)&&(fabs(aP1.y-1894.23)<0.1))
                       ||((fabs(aP2.x-494.410)<0.1)&&(fabs(aP2.y-1894.23)<0.1)))
                    {
                        cout<<aNameIn1<<endl;
                        std::cout<<"For pair : "<<aP1<<" "<<aP2<<std::endl;
                        std::cout<<"Result Homol: "<<aPointOnPic1<<" "<<aPointOnPic2<<std::endl;
                        if (aPointOnPic1) aPointOnPic1->getHomol()->print();
                        if (aPointOnPic2) aPointOnPic2->getHomol()->print();
                    }*/
                    /*if ((aPointOnPic1 && (aPointOnPic1->getHomol()->getId()==7904))||
                        (aPointOnPic2 && (aPointOnPic2->getHomol()->getId()==7904)))
                    {
                        std::cout<<"For pair : "<<aP1<<" "<<aP2<<std::endl;
                        std::cout<<"Result Homol: "<<aPointOnPic1<<" "<<aPointOnPic2<<std::endl;
                        if (aPointOnPic1) aPointOnPic1->getHomol()->print();
                        if (aPointOnPic2) aPointOnPic2->getHomol()->print();
                    }*/

                    if (aPointOnPic1 && (!aPointOnPic2))
                    {
                        aPointOnPic1->getHomol()->add(pic2,aP2);
                        aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if (aPointOnPic2 && (!aPointOnPic1))
                    {
                        aPointOnPic2->getHomol()->add(pic1,aP1);
                        aPointOnPic2->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if (aPointOnPic1 && aPointOnPic2 &&(aPointOnPic1->getHomol()!=aPointOnPic2->getHomol()))
                    {
                        cPointOnPic * aPointOnPic12=aPointOnPic1->getHomol()->getPointOnPic(pic2);
                        cPointOnPic * aPointOnPic21=aPointOnPic2->getHomol()->getPointOnPic(pic1);
                        if (
                            ((aPointOnPic21)
                              &&(
                                 (fabs(aPointOnPic21->getPt().x-aP2.x)>0.1)
                                 ||(fabs(aPointOnPic21->getPt().y-aP2.y)>0.1))
                                )
                            ||
                            ((aPointOnPic12)
                              &&(
                                 (fabs(aPointOnPic12->getPt().x-aP1.x)>0.1)
                                 ||(fabs(aPointOnPic12->getPt().y-aP1.y)>0.1))
                                )
                           )
                        {
                            //std::cout<<"Bad homols!\n";
                            aPointOnPic1->getHomol()->setBad();
                            aPointOnPic2->getHomol()->setBad();
                            //aPointOnPic1->getHomol()->print();
                            //aPointOnPic2->getHomol()->print();
                        }
                        
                        //merge the two homol points
                        aPointOnPic1->getHomol()->add(aPointOnPic2->getHomol());
                        aPointOnPic2->getHomol()->getPointOnPics().clear();
                        //don't remove the homol, useless
                        /*std::list<cHomol*>::iterator itHomol;
                        for (itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
                        {
                            if ((*itHomol)==aPointOnPic2->getHomol())
                            {
                                allHomolsIn.erase(itHomol);
                                break;
                            }
                        }*/
                        aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if ((!aPointOnPic1) && (!aPointOnPic2))
                    {
                        //new homol point
                        allHomolsIn.push_back(new cHomol);
                        allHomolsIn.back()->add(pic1,aP1);
                        allHomolsIn.back()->add(pic2,aP2);
                        allHomolsIn.back()->addAppearsOnCouple(pic1,pic2);
                        
                        /*if (((fabs(aP1.x-494.410)<0.1)&&(fabs(aP1.y-1894.23)<0.1))
                           ||((fabs(aP2.x-494.410)<0.1)&&(fabs(aP2.y-1894.23)<0.1)))
                        {
                            cout<<aNameIn1<<endl;
                            allHomolsIn.back()->print();
                        }*/
                    
                    }else if (aPointOnPic1 && aPointOnPic2 &&(aPointOnPic1->getHomol()==aPointOnPic2->getHomol()))
                    {
                        aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    
                    
                }
            }
        }
    }

    ELISE_ASSERT(allHomolsIn.size()>0,"ERROR: No pack found!");
    
    
    /*cout<<"Cleaning Homol list..."<<std::endl;
    for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();)
    {
        if ((*itHomol)->isBad())
            allHomolsIn.erase(itHomol);
        else
            itHomol++;
    }*/

    /*if (false)//check both ways
    {
        //check if homol apear everywhere they should
        cout<<"Checking Homol both ways..";
        //for every homol
        int nbInconsistantHomol=0;
        for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
        {
            cHomol *aHomol=(*itHomol);
            cPic *aPic1=0;
            cPic *aPic2=0;
            for (unsigned int i=0;i<aHomol->getAppearsOnCoupleSize();i++)
            {
                aPic1=aHomol->getAppearsOnCoupleA(i);
                aPic2=aHomol->getAppearsOnCoupleB(i);
                if (!aHomol->appearsOnCouple2way(aPic1,aPic2))
                {
                    aHomol->setBad();
                    i=aHomol->getPointOnPics().size();//end second loop
                    nbInconsistantHomol++;
                    break;
                }
            }
            if ((aHomol->getId()%1000)==0) cout<<"."<<flush;
        }
        std::cout<<"Done.\n"<<nbInconsistantHomol<<" inconsistant homols found."<<endl;
    }*/

    if (veryStrict)
    {
        //check if homol apear everywhere they should
        cout<<"Checking Homol integrity..";
        //for every homol
        int nbInconsistantHomol=0;
        for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
        {
            cHomol *aHomol=(*itHomol);
            cPic *aPic1=0;
            cPic *aPic2=0;
            #ifdef ReductHomolImage_VeryStrict_DEBUG
            cout<<"For ";
            aHomol->print();
            #endif
            //for every combination of PointOnPic
            for (unsigned int i=0;i<aHomol->getPointOnPics().size();i++)
            {
                aPic1=aHomol->getPointOnPics()[i]->getPic();
                for (unsigned int j=i+1;j<aHomol->getPointOnPics().size();j++)
                {
                    //if the pack exist
                    aPic2=aHomol->getPointOnPics()[j]->getPic();
                    //std::string aNameIn = aDirImages + aICNM->Assoc1To2(aKHIn,aPic1->getName(),aPic2->getName(),true);
                    std::string aNameIn=aCKin.get(aPic1->getName(),aPic2->getName());
                    if (ELISE_fp::exist_file(aNameIn))
                    {
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        cout<<"   "<<aNameIn<<": ";
                        #endif
                        //check that homol has been seen in this couple of pictures
                        if (!aHomol->appearsOnCouple2way(aPic1,aPic2))
                        {
                            #ifdef ReductHomolImage_VeryStrict_DEBUG
                            cout<<"No!\n";
                            #endif
                            aHomol->setBad();
                            i=aHomol->getPointOnPics().size();//end second loop
                            nbInconsistantHomol++;
                            break;
                        }
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        else cout<<"OK!\n";
                        #endif

                    }
                }
                if (aHomol->isBad()) break;
            }
            if ((aHomol->getId()%1000)==0) cout<<"."<<flush;
        }
        std::cout<<"Done.\n"<<nbInconsistantHomol<<" inconsistant homols found."<<endl;
    }
    
    int aNumBadHomol=0;
    for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
    {
        if ((*itHomol)->isBad()) aNumBadHomol++;
    }

    std::cout<<"Found "<<allHomolsIn.size()<<" Homol points (incl. "<<aNumBadHomol<<" bad ones): "<<100*aNumBadHomol/allHomolsIn.size()<<"% bad!\n";

    /*cout<<"Cleaning Homol list..."<<std::endl;
    for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();)
    {
        if ((*itHomol)->isBad())
            allHomolsIn.erase(itHomol);
        else
            itHomol++;
    }*/


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
    //std::sort(allPics.begin(), allPics.end(), compareNumberOfHomolPics);
    
    #ifdef ReductHomolImage_DEBUG
    std::cout<<"Homols per image:";
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic=(*itPic1).second;
        aPic->printHomols();
        std::cout<<std::endl<<"  - "<<aPic->getName()<<" "<<std::flush;
        /*std::list<cPointOnPic*>::iterator itHomolPoint;
        for (itHomolPoint=aPic->getAllPointsOnPic()->begin();
             itHomolPoint!=aPic->getAllPointsOnPic()->end();
             ++itHomolPoint)
        {
            std::cout<<(*itHomolPoint)->getHomol()<<" "<<std::flush;
            std::cout<<(*itHomolPoint)->getHomol()->getId()<<" "<<std::flush;
        }*/
    }
    std::cout<<std::endl;
    #endif

    /*std::cout<<"Search for particular homol:\n";
    std::list<cHomol*>::iterator itHomol;
    std::vector<cPointOnPic*>::iterator itPointOnPic;
    for (itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
    {
        for (itPointOnPic=(*itHomol)->getPointOnPics().begin();
             itPointOnPic!=(*itHomol)->getPointOnPics().end();
             ++itPointOnPic)
        {
            if (((*itPointOnPic)->getPt().x==4695.720000)
                    &&((*itPointOnPic)->getPt().y==1305.77))
            {
                (*itHomol)->print();
            }
        }
    }*/



    
    //create new homols ------------------------------------------------
    std::cout<<"Create new homol..";
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic=(*itPic1).second;
        std::cout<<"."<<flush;
        //std::cout<<"  "<<aPic->getName()<<endl;
        aPic->selectHomols();
    }
    std::cout<<"Done!"<<endl;

    #ifdef ReductHomolImage_DEBUG
    std::cout<<"New Homols per image:";
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic=(*itPic1).second;
        std::cout<<std::endl<<"  - "<<aPic->getName()<<" "<<std::flush;
        /*std::list<cPointOnPic*>::iterator itPointsOnPic;
        for (itPointsOnPic=aPic->getAllSelectedPointsOnPic()->begin();
             itPointsOnPic!=aPic->getAllSelectedPointsOnPic()->end();
             ++itPointsOnPic)
        {
            std::cout<<(*itPointsOnPic)->getHomol()->getId()<<" "<<std::flush;
        }*/
    }
    std::cout<<std::endl;
    #endif

    /*
    cPic *aPic=allPics[4];
    std::cout<<"Homol init sur "<<aPic->getName()<<":\n";
    std::list<cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=aPic->getAllPointsOnPic()->begin();
         itPointOnPic!=aPic->getAllPointsOnPic()->end();
         ++itPointOnPic)
    {
        (*itPointOnPic)->getHomol()->print();
    }
    std::cout<<"Homol select sur "<<aPic->getName()<<":\n";
    //std::list<cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=aPic->getAllSelectedPointsOnPic()->begin();
         itPointOnPic!=aPic->getAllSelectedPointsOnPic()->end();
         ++itPointOnPic)
    {
        (*itPointOnPic)->getHomol()->print();
    }
    */



    std::cout<<"Write new Packs:\n";
    std::ofstream aFileBadPictureNames;
    aFileBadPictureNames.open(aPoubelleName.c_str());
    if (!aFileBadPictureNames.is_open())
    {
        std::cout<<"Impossible to create \""<<aPoubelleName<<"\" file!\n";
        return -1;
    }
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* pic1=(*itPic1).second;
        std::cout<<" - "<<pic1->getName()<<": "<<pic1->getPercentWinUsed()<<"% of the picture covered ("<<pic1->getAllSelectedPointsOnPicSize()<<" points)\n";
        if (pic1->getPercentWinUsed()<25)
            aFileBadPictureNames<<pic1->getName()<<"\n";
        for (itPic2=itPic1;itPic2!=allPics.end();++itPic2)
        {
            if (itPic2==itPic1) continue; //with c++11: itPic2=next(itPic1)
            cPic* pic2=(*itPic2).second;
            //using aICNM->Assoc1To2 is needed to create directories!
            std::string aNameOut1 = aDirImages + aICNM->Assoc1To2(aKHOut,pic1->getName(),pic2->getName(),true);
            //std::string aNameOut1 = aDirImages + aCKout.get(pic1->getName(),pic2->getName());
            std::string aNameOut2 = aDirImages + aICNM->Assoc1To2(aKHOut,pic2->getName(),pic1->getName(),true);
            //std::string aNameOut2 = aDirImages + aCKout.get(pic2->getName(),pic1->getName());
            //std::cout<<"For "<<aNameOut1<<" and "<<aNameOut2<<": "<<endl;
            
            pic1->fillPackHomol(pic2,aNameOut1,aNameOut2);
        }
    }
    aFileBadPictureNames.close();

    std::cout<<"You can look at \""<<aPoubelleName<<"\" for a list of suspicious pictures.\n";
  
   
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
