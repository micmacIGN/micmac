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

#include "schnaps.h"

#include "StdAfx.h"
#include <fstream>
#include <algorithm>
#include <iterator>

int cPicSize::mTargetNumWindows=-1;//make an error if not initialized
std::vector<cPicSize> cPic::mAllSizes;


cPicSize::cPicSize(Pt2di aSz) :
    mPicSz(aSz)
{
    float aXYratio=((float)aSz.x)/aSz.y;
    mNbWin.x=sqrt((double)mTargetNumWindows)*sqrt(aXYratio)+1;
    mNbWin.y=sqrt((double)mTargetNumWindows)/sqrt(aXYratio)+1;
    mWinSz.x=((float)mPicSz.x)/mNbWin.x+0.5;
    mWinSz.y=((float)mPicSz.y)/mNbWin.y+0.5;
    mUsageBuffer=mNbWin.x/10;//where the arbitrary buffer size is calculated
    //std::cout<<"Size constr: "<<this<<"   "<<aSz<<" => "<<mNbWin<<std::endl;
}

int cPointOnPic::mPointOnPicCounter=0;

int cHomol::mHomolCounter=0;



//----------------------------------------------------------------------------

cPointOnPic::cPointOnPic() :
    mPic(0),mPt(),mHomol(0)
{
    cout<<"cPointOnPic::cPointOnPic()\n";
}

cPointOnPic::cPointOnPic(const cPointOnPic &o) :
    mId(o.mId),mPic(o.mPic),mPt(o.mPt),mHomol(o.mHomol)
{
    cout<<"cPointOnPic::cPointOnPic(cPointOnPic &o) "<<mId<<"   "<<&o<<" => "<<this<<"\n";
}


cPointOnPic::cPointOnPic(cPic *aPic,Pt2dr aPt,cHomol* aHomol) :
    mPic(aPic),mPt(aPt),mHomol(aHomol)
{
    mId=mPointOnPicCounter;
    mPointOnPicCounter++;

#ifdef ReductHomolImage_DEBUG
    cout<<"New cPointOnPic "<<this<<" for Homol "<<mHomol->getId()<<"   homol "<<mHomol<<endl;
#endif
}

void cPointOnPic::print()
{
    cout<<"     cPointOnPic "<<mId<<": "<<this<<" on "<<mPic->getName()<<": "<<mPt<<"   homol "<<mHomol<<endl;
}

cPointOnPic * cPic::findPointOnPic(Pt2dr & aPt)
{
    std::map<double,cPointOnPic*>::iterator it;
    it=mAllPointsOnPic.find(makePOPKey(aPt));
    return (it==mAllPointsOnPic.end()) ? 0 : it->second;
}


cPointOnPic* cHomol::getPointOnPic(cPic * aPic)
{
    for (unsigned int i=0;i<mPointOnPics.size();i++)
        if (getPointOnPic(i)->getPic()==aPic)
            return getPointOnPic(i);
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
    for (unsigned int i=0;i<mPointOnPics.size();i++)
    {
        if (mPointOnPics[i]->getPic()==aPic)
        {
            //std::cout<<"Bad Homol!\n";
            setBad();
            return false;
        }
    }

    mPointOnPics.push_back(new cPointOnPic(aPic,aPt,this));
    aPic->addPointOnPic(getPointOnPic(getPointOnPicsSize()-1));
    
    return true;
}

void cHomol::add(cHomol *aHomol)
{
#ifdef ReductHomolImage_DEBUG
    std::cout<<"Merge Homol "<<mId<<" and "<<aHomol->mId<<"\n";
#endif
    for (unsigned int i=0;i<aHomol->getPointOnPicsSize();i++)
    {
        mPointOnPics.push_back((aHomol->getPointOnPic(i)));
        mPointOnPics.back()->setHomol(this);
        aHomol->getPointOnPic(i)->getPic()->removeHomolPoint(aHomol->getPointOnPic(i));
        aHomol->getPointOnPic(i)->getPic()->addPointOnPic(getPointOnPic(getPointOnPicsSize()-1));
    }
    for (unsigned int i=0;i<aHomol->mAppearsOnCoupleA.size();i++)
    {
        mAppearsOnCoupleA.push_back(aHomol->mAppearsOnCoupleA[i]);
        mAppearsOnCoupleB.push_back(aHomol->mAppearsOnCoupleB[i]);
    }
    aHomol->mPointOnPics.clear();
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

bool cHomol::appearsOnCouple1way(cPic * aPicA,cPic * aPicB)
{
    for (unsigned int i=0;i<mAppearsOnCoupleA.size();i++)
    {
        //must be seen on both directions
        if ((mAppearsOnCoupleA[i]==aPicA)&&(mAppearsOnCoupleB[i]==aPicB))
        {
            return true;
        }
        if ((mAppearsOnCoupleA[i]==aPicB)&&(mAppearsOnCoupleB[i]==aPicA))
        {
            return true;
        }
    }
    return false;
}



//checks if the same picture is not in both Homol
bool cHomol::checkMerge(cHomol* aHomol)
{
    for (unsigned int i=0;i<mPointOnPics.size();i++)
    {
        if (aHomol->getPointOnPic(getPointOnPic(i)->getPic()))
            return false;
    }
    return true;
}

//----------------------------------------------------------------------------

long cPic::mNbIm=0;

cPic::cPic(std::string aDir, std::string aName) :
    mName(aName),mPicSize(0),mId(mNbIm++)//,mNbWinUsed(0)
{
    Tiff_Im aPic( Tiff_Im::StdConvGen(aDir+"/"+aName,1,false)); //to read file in Tmp-MM-Dir if needed
    Pt2di aPicSize=aPic.sz();
    bool found=false;
    for (unsigned int i=0;i<mAllSizes.size();i++)
      if (mAllSizes[i].getPicSz()==aPicSize)
      {
        found=true;
        mPicSize=&(mAllSizes[i]);
        break;
      }
    if (!found)
    {
      mAllSizes.push_back(cPicSize(aPicSize));
      mPicSize=&(mAllSizes.back());
    }
    //cout<<"Pic windows: "<<mPicSize->getNbWin().x<<" "<<mPicSize->getNbWin().y<<endl;
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
                 <<" multi "<<(itPointOnPic->second)->getHomol()->getPointOnPicsSize()<<std::endl;
    }
}

//Luc G. : Here and next function, std::make_pair<double,cPointOnPic*> does not compile on VS15 (CPP11),
// make_pair should not be used with typedefinition, because it's advantage over std::pair is that you don't have to do so (apparently).
void cPic::addPointOnPic(cPointOnPic* aPointOnPic)
{
    mAllPointsOnPic.insert(
        std::make_pair(
            makePOPKey(aPointOnPic->getPt()),
            aPointOnPic));
}

bool cPic::addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic)
{
    std::map<double,cPointOnPic*>::iterator it;
    it=mAllSelectedPointsOnPic.find(makePOPKey(aPointOnPic->getPt()));
    if (it!=mAllSelectedPointsOnPic.end())
        return false;
    else
        mAllSelectedPointsOnPic.insert(
            std::make_pair(
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

float cPic::getPercentWinUsed(int nbWin)
{
    //return 100.0*((float)mNbWinUsed)/(mPicSize->getNbWin().x*mPicSize->getNbWin().y);
    int nbWinUsed=0;
    for (int i=0;i<mPicSize->getNbWin().x*mPicSize->getNbWin().y;i++)
        if (mWinUsed[i]) nbWinUsed++;
    if (getAllSelectedPointsOnPicSize()<nbWin)
        nbWinUsed*=(sqrt((float)(getAllSelectedPointsOnPicSize())/nbWin));
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
    //cout<<"On pic "<<getName()<<"\n";
    //cout<<"Pic windows: "<<mPicSize->getNbWin().x<<" "<<mPicSize->getNbWin().y<<endl;
    std::vector< std::vector<cPointOnPic*> > winBestPoP;
    std::vector< std::vector<unsigned int> > winBestMulti;
    winBestPoP.resize(getPicSize()->getNbWin().y);
    for (unsigned int i=0;i<winBestPoP.size();i++)
        winBestPoP[i].resize(getPicSize()->getNbWin().x,0);
    winBestMulti.resize(getPicSize()->getNbWin().y);
    for (unsigned int i=0;i<winBestMulti.size();i++)
        winBestMulti[i].resize(getPicSize()->getNbWin().x,0);
    
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
        if (winBestMulti[y][x]<aPoP->getHomol()->getPointOnPicsSize())
        {
            winBestMulti[y][x]=aPoP->getHomol()->getPointOnPicsSize();
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

        if (winBestMulti[y][x]<aPoP->getHomol()->getPointOnPicsSize()) //plante ici
        {
            winBestMulti[y][x]=aPoP->getHomol()->getPointOnPicsSize();
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
                cHomol * aHomol=aBestSelectedPointOnPic->getHomol();
                
                for (unsigned int i=0;i<aHomol->getPointOnPicsSize();i++)
                {
                    cPic * aOtherPic=aHomol->getPointOnPic(i)->getPic();
                    aOtherPic->addSelectedPointOnPicUnique(aHomol->getPointOnPic(i));
                }
            }
            if (winBestMulti[y][x]>0)
                setWinUsed(x,y);
        }
    }
}


void cPic::selectAllHomols()
{
    mAllSelectedPointsOnPic=mAllPointsOnPic;
}


void cPic::fillPackHomol(cPic* aPic2,string & aDirImages,cInterfChantierNameManipulateur * aICNM,std::string & aKHOut)
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
    
    if (aPackOut1.size()>3)
    {
        std::string aNameOut1 = aDirImages + aICNM->Assoc1To2(aKHOut,getName(),aPic2->getName(),true);
        std::string aNameOut2 = aDirImages + aICNM->Assoc1To2(aKHOut,aPic2->getName(),getName(),true);
        //std::cout<<aNameOut1<<": "<<aPackOut1.size()<<" pairs."<<endl;
        //std::cout<<aNameOut2<<": "<<aPackOut2.size()<<" pairs."<<endl;
        aPackOut1.StdPutInFile(aNameOut1);
        aPackOut2.StdPutInFile(aNameOut2);
    }
    
}

std::vector<int> cPic::getStats(bool before)
{
    std::vector<int> numHomolMultiplicity;
    //homol-0: 0, homol-1: 0,homol-2: ??

    std::map<double,cPointOnPic*> *pointsOnPic=before?&mAllPointsOnPic:&mAllSelectedPointsOnPic;
    std::map<double,cPointOnPic*>::iterator itPointOnPic;
    for (itPointOnPic=pointsOnPic->begin();itPointOnPic!=pointsOnPic->end();++itPointOnPic)
    {
        if ((itPointOnPic->second)->getHomol()->isBad()) continue;
        unsigned int multi=(itPointOnPic->second)->getHomol()->getPointOnPicsSize();
        if (multi>=numHomolMultiplicity.size())
            numHomolMultiplicity.resize(multi+1,0);
        numHomolMultiplicity[multi]++;

        //if (multi>4)
        //    (itPointOnPic->second)->getHomol()->print();
    }
    return numHomolMultiplicity;
}

bool compareNumberOfHomolPics (cPic* aPic1,cPic* aPic2)
{
    return (aPic1->getAllPointsOnPicSize()<aPic2->getAllPointsOnPicSize());
}


//----------------------------------------------------------------------------

CompiledKey2::CompiledKey2(cInterfChantierNameManipulateur * aICNM,std::string aKH):
    mICNM(aICNM),mKH(aKH)
{
    std::string aNameIn = mICNM->Assoc1To2(aKH,"X1X","X2X",true);
    std::size_t pos1 = aNameIn.find("X1X");
    std::size_t pos2 = aNameIn.find("X2X");
    mPart1 = aNameIn.substr (0,pos1);
    mPart2 = aNameIn.substr (pos1+3,pos2-pos1-3);
    mPart3 = aNameIn.substr (pos2+3);

    //remove the created folder "X1X":
    ELISE_fp::RmDir(mPart1+"X1X");
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


void computeAllHomol(std::string aDirImages,
                     std::string aPatIm,
                     const std::vector<std::string> &aSetIm,
                     std::list<cHomol> &allHomolsIn,
                     CompiledKey2 &aCKin,
                     std::map<std::string,cPic*> &allPics,
                     bool veryStrict,
                     int aNumWindows)
{
    cPicSize::mTargetNumWindows=aNumWindows;
    for (unsigned int i=0;i<aSetIm.size();i++)
        allPics.insert(std::make_pair<std::string,cPic*>(aSetIm[i]+aCKin.getSuffix(),new cPic(aDirImages,aSetIm[i])));

    ELISE_ASSERT(aSetIm.size()>0,"ERROR: No image found!");

    std::cout<<"All sizes: \n";
    for (unsigned int i=0;i<cPic::getAllSizes()->size();i++)
    {
        std::cout<<"  * "<<cPic::getAllSizes()->at(i).getPicSz()
                 <<" => "<<cPic::getAllSizes()->at(i).getNbWin()
                 <<" windows of "<<cPic::getAllSizes()->at(i).getWinSz()<<" pixels"<<endl;
    }

    //read all homol points --------------------------------------------
    std::cout<<"Read packs of homol points:"<<endl;

    std::map<std::string,cPic*>::iterator itPic1,itPic2;
    int i=0;
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    //for (unsigned int i=0;i<allPics.size();i++)
    {
        cPic *pic1=(*itPic1).second;//allPics[i];
        std::cout<<" Picture "<<pic1->getName()<<": ";
        //get all pictures having pac with pic1

        if (!ELISE_fp::IsDirectory(aCKin.getDir(pic1->getName(),aPatIm)))
        {
            std::cout<<"No homol file."<<endl;
            continue;
        }

        cInterfChantierNameManipulateur * homolICNM=cInterfChantierNameManipulateur::BasicAlloc(aCKin.getDir(pic1->getName(),aPatIm));
        std::list<std::string> aSetPac = homolICNM->StdGetListOfFile(aCKin.getFile(pic1->getName(),aPatIm),2,false);

        std::cout<<"Found "<<aSetPac.size()<<" homol files and ";

        i++;
        std::list<std::string>::iterator itPacName;
        long nb_homol_raw=0;
        for (itPacName=aSetPac.begin();itPacName!=aSetPac.end();++itPacName)
        {
            itPic2=allPics.find( (*itPacName) );
            if (itPic2==allPics.end()) continue; //if the pic has been removed after Tapioca
            cPic *pic2=(*itPic2).second;

            std::string aNameIn1= aDirImages + aCKin.get(pic1->getName(),pic2->getName());
            //if (ELISE_fp::exist_file(aNameIn1))
            {
                ElPackHomologue aPackIn1 =  ElPackHomologue::FromFile(aNameIn1);
                //cout<<aNameIn1<<"  Pack size: "<<aPackIn1.size()<<"\n";
                nb_homol_raw+=aPackIn1.size();
                pic1->getNbRawLinks()->insert(std::make_pair<cPic*&, long>(pic2,aPackIn1.size()));
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
                    /*if ((aPointOnPic1 && (aPointOnPic1->getHomol()->getId()==8897))||
                        (aPointOnPic2 && (aPointOnPic2->getHomol()->getId()==8897)))
                    {
                        std::cout<<"For pair : "<<aP1<<" "<<aP2<<std::endl;
                        std::cout<<"Result Homol: "<<aPointOnPic1<<" "<<aPointOnPic2<<std::endl;
                        if (aPointOnPic1) aPointOnPic1->getHomol()->print();
                        if (aPointOnPic2) aPointOnPic2->getHomol()->print();
                    }*/

                    if (aPointOnPic1 && (!aPointOnPic2)) // added on pic1 but haven't added on pic 2
                    {
                        aPointOnPic1->getHomol()->add(pic2,aP2);
                        if (veryStrict) aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if (aPointOnPic2 && (!aPointOnPic1))
                    {
                        aPointOnPic2->getHomol()->add(pic1,aP1);
                        if (veryStrict) aPointOnPic2->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if (aPointOnPic1 && aPointOnPic2 &&(aPointOnPic1->getHomol()!=aPointOnPic2->getHomol()))
                    {
                        if (
                                !(aPointOnPic1->getHomol()->checkMerge(aPointOnPic2->getHomol()))
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
                        if (veryStrict) aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }
                    else if ((!aPointOnPic1) && (!aPointOnPic2))
                    {
                        //new homol point
                        allHomolsIn.push_back(cHomol());
                        allHomolsIn.back().add(pic1,aP1);
                        allHomolsIn.back().add(pic2,aP2);
                        if (veryStrict) allHomolsIn.back().addAppearsOnCouple(pic1,pic2);

                    }else if (aPointOnPic1 && aPointOnPic2 &&(aPointOnPic1->getHomol()==aPointOnPic2->getHomol()))
                    {
                        if (veryStrict) aPointOnPic1->getHomol()->addAppearsOnCouple(pic1,pic2);
                    }


                }
            }
        }
        std::cout<<nb_homol_raw<<" raw homol couples."<<std::endl;
    }

    ELISE_ASSERT(allHomolsIn.size()>0,"ERROR: No homol file found!");

}

void networkExport(std::map<std::string,cPic*> &allPics, int aFactPH)
{
    std::ostringstream oss;
    oss<<"var nodes = [\n";
    std::map<std::string,cPic*>::iterator itPic1;
    std::map<cPic*,long>::iterator itPic2;
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic=(*itPic1).second;
        oss<<"  {id: "<<aPic->getId()<<", 'label': '"<<aPic->getName()<<"', 'group': 1},\n";
    }
    oss<<"];\n";
    oss<<"var edges = [\n";
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic1=(*itPic1).second;
        for (itPic2=aPic1->getNbRawLinks()->begin();itPic2!=aPic1->getNbRawLinks()->end();++itPic2)
        {
            cPic* aPic2=(*itPic2).first;
            if ((*itPic2).second>=10)
                oss<<"  {'from': "<<aPic1->getId()<<", 'to': "<<aPic2->getId()<<", value: "<<1+(*itPic2).second/aFactPH<<"},\n";
        }
    }
    oss<<"];\n";

    std::cout<<"To display network, use micmac_Documentation/NEW-DATA/schnaps_disp_graph/disp_graph.html\n";

    std::ofstream aNetworkfile;
    aNetworkfile.open("data.js");
    if (!aNetworkfile.is_open())
    {
        std::cout<<"Impossible to create \""<<"data.js"<<"\" file!\n";
        return;
    }
    aNetworkfile<<oss.str();
    aNetworkfile.close();
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
    bool doShowStats=false;
    bool ExeWrite=true;
    bool DoNotFilter=false;
    double aMinPercentCoverage=30;//if %coverage<aMinPercentCoverage, add to poubelle!
    bool aMove=false;//if true, move poubelle images to a folder named "Poubelle/"
    int aMinimalMultiplicity=1;
    std::string aNameTrashFolder = "";
    bool aNetworkExport=false;//export html network image
    int aFactPH(10);

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
                   << EAM(ExeWrite,"ExeWrite",true,"Execute write output homol dir, def=true",eSAM_InternalUse)
                   << EAM(aOutHomolDirName, "HomolOut", true, "Output Homol directory suffix (default: _mini)")
                   << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                   << EAM(veryStrict,"VeryStrict",true,"Be very strict with homols (remove any suspect), def=false")
                   << EAM(doShowStats,"ShowStats",true,"Show Homol points stats before and after filtering, def=false")
                   << EAM(DoNotFilter,"DoNotFilter",true,"Write homol after recomposition, without filterning, def=false")
                   << EAM(aPoubelleName,"PoubelleName",true,string("Where to write suspicious pictures names, def=\"")+aPoubelleName+"\"")
                   << EAM(aMinPercentCoverage,"minPercentCoverage",true,"Minimum % of coverage to avoid adding to poubelle, def=30")
                   << EAM(aMove,"MoveBadImgs",true,"Move bad images to a trash folder called Poubelle, Def=false")
                   << EAM(aNameTrashFolder,"OutTrash",true,"Output name of trash folder if moving bad images, Def=Poubelle")
                   << EAM(aMinimalMultiplicity,"MiniMulti",true,"Minimal Multiplicity of selected points, Def=1")
                   << EAM(aNetworkExport,"NetworkExport",true,"Export Network (in js), Def=false")
                   << EAM(aFactPH,"DivPH",true,"in exported network, denominator to decrease the number of tie point which is used for displaying strength of a relation between 2 images, def 10.")
      );

    if (MMVisualMode) return EXIT_SUCCESS;
    
    if(aNameTrashFolder == "")
    {
		aNameTrashFolder = "Poubelle";
	}


    std::cout<<"Number of searching windows: "<<aNumWindows<<std::endl;

    // Initialize name manipulator & files
    std::string aDirXML,aDirImages,aPatIm;
    //std::string aGCPFileTmpName;
    //SplitDirAndFile(aDirXML,aGCPFileTmpName,aGCPFileName);
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;

    StdCorrecNameHomol(aInHomolDirName,aDirImages);

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


    // Init Keys for homol files
    std::list<cHomol> allHomolsIn;
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

    std::cout<<"Found "<<aSetIm.size()<<" pictures."<<endl;

    computeAllHomol(aDirImages,aPatIm,aSetIm,allHomolsIn,aCKin,allPics,veryStrict,aNumWindows);

    if (aNetworkExport)
        networkExport(allPics,aFactPH);
    
    /*cout<<"Cleaning Homol list..."<<std::endl;
    for (std::list<cHomol*>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();)
    {
        if ((*itHomol)->isBad())
            allHomolsIn.erase(itHomol);
        else
            itHomol++;
    }*/

    if (aMinimalMultiplicity>1)
    {
        for (std::list<cHomol>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
        {
            cHomol &aHomol=(*itHomol);
            if (aHomol.getPointOnPicsSize()<(unsigned)aMinimalMultiplicity)
                aHomol.setBad();
        }
    }

    if (veryStrict)
    {
        //check if homol apear everywhere they should
        cout<<"Checking Homol integrity..";
        //for every homol
        int nbInconsistantHomol=0;
        for (std::list<cHomol>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
        {
            cHomol &aHomol=(*itHomol);
            cPic *aPic1=0;
            cPic *aPic2=0;
            #ifdef ReductHomolImage_VeryStrict_DEBUG
            cout<<"For ";
            aHomol.print();
            #endif
            //for every combination of PointOnPic
            for (unsigned int i=0;i<aHomol.getPointOnPicsSize();i++)
            {
                aPic1=aHomol.getPointOnPic(i)->getPic();
                for (unsigned int j=i+1;j<aHomol.getPointOnPicsSize();j++)
                {
                    //if the pack exist
                    aPic2=aHomol.getPointOnPic(j)->getPic();
                    //std::string aNameIn = aDirImages + aICNM->Assoc1To2(aKHIn,aPic1->getName(),aPic2->getName(),true);
                    std::string aNameIn=aCKin.get(aPic1->getName(),aPic2->getName());
                    if (ELISE_fp::exist_file(aNameIn))
                    {
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        cout<<"   "<<aNameIn<<": ";
                        #endif
                        //check that homol has been seen in this couple of pictures
                        if (!aHomol.appearsOnCouple1way(aPic1,aPic2))
                        {
                            #ifdef ReductHomolImage_VeryStrict_DEBUG
                            cout<<"No!\n";
                            #endif
                            aHomol.setBad();
                            i=aHomol.getPointOnPicsSize();//end second loop
                            nbInconsistantHomol++;
                            break;
                        }
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        else cout<<"OK!\n";
                        #endif

                    }
                }
                if (aHomol.isBad()) break;
            }
            if ((aHomol.getId()%1000)==0) cout<<"."<<flush;
        }
        std::cout<<"Done.\n"<<nbInconsistantHomol<<" inconsistant homols found."<<endl;
    }
    
    int aNumBadHomol=0;
    for (std::list<cHomol>::iterator itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
    {
        if ((itHomol)->isBad()) aNumBadHomol++;
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
    std::list<cHomol>::iterator itHomol;
    for (itHomol=allHomolsIn.begin();itHomol!=allHomolsIn.end();++itHomol)
    {
        //std::cout<<(*itHomol)->getPointOnPics().size()<<" ";
        itHomol->print();
    }
    std::cout<<std::endl;
    #endif
    
    //sort pics on number of homols ------------------------------------
    //std::sort(allPics.begin(), allPics.end(), compareNumberOfHomolPics);
    std::map<std::string,cPic*>::iterator itPic1,itPic2;

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

    if (!DoNotFilter)
    {
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
    }else{
        for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
        {
            cPic* aPic=(*itPic1).second;
            aPic->selectAllHomols();
        }
    }

    int nbBadPictures=0;
    if (ExeWrite)
    {
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
            std::cout<<" - "<<pic1->getName()<<": "<<pic1->getPercentWinUsed(aNumWindows)<<"% of the picture covered ("<<pic1->getAllSelectedPointsOnPicSize()<<" points)";
            if (pic1->getPercentWinUsed(aNumWindows)<aMinPercentCoverage)
            {
                nbBadPictures++;
                aFileBadPictureNames<<pic1->getName()<<"\n";
                cout<<" rejected!";
                if(aMove)
                {
					ELISE_fp::MkDirSvp(aNameTrashFolder); //create folder if does not exist
					ELISE_fp::MvFile(pic1->getName(),aNameTrashFolder);cout<<"\n"; //move it to poubelle folder
					cout<< " moved to "<<aNameTrashFolder<<"\n";
				}
            }
            std::cout<<std::endl;
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

                pic1->fillPackHomol(pic2,aDirImages,aICNM,aKHOut);
            }
        }
        aFileBadPictureNames.close();
    }

    if (doShowStats)
    {
        for (int j=0;j<2;j++)
        {
            if (j==0)
                std::cout<<"\nStats BEFORE filtering:\n";
            else
                std::cout<<"\nStats AFTER filtering:\n";
            for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
            {
                std::cout<<itPic1->second->getName()<<": ";
                std::vector<int> homolStats=itPic1->second->getStats(j==0);
                for (unsigned int i=2;i<homolStats.size();i++)
                {
                    std::cout<<i<<":"<<homolStats[i]<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }

    if (ExeWrite)
    {
        std::cout<<nbBadPictures<<" pictures rejected."<<std::endl;
        std::cout<<"\nYou can look at \""<<aPoubelleName<<"\" for a list of suspicious pictures.\n";
    }
  
    //cleaning
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
        delete itPic1->second;
    allPics.clear();
   
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
