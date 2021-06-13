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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"



/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/


     // ============== cCaseX11Xml =======================================


void cCaseX11Xml::string(int aPos,const std::string & aName)
{
   mW.fixed_string_middle(mBox,aPos,aName,mW.pdisc()(P8COL::black),true);
}

Pt2di cCaseX11Xml::P0Line() 
{
   return Pt2di (mBox._p0.x+2,mBox._p1.y-3);
}

void cCaseX11Xml::Efface(Col_Pal aCoul)
{
      mW.fill_rect(Pt2dr(mBox._p0),Pt2dr(mBox._p1),aCoul);
}
void cCaseX11Xml::Efface(int aCoul)
{
   if (aCoul >=0)
      Efface(mW.pdisc()(aCoul));
}
void cCaseX11Xml::Efface() {Efface(mCoul);}

cCaseX11Xml * cCaseX11Xml::Alloc(Video_Win aW,Box2di aBox,int aCoul)
{
   return new cCaseX11Xml(aW,aBox,aCoul);
}

cCaseX11Xml::cCaseX11Xml(Video_Win aW,Box2di aBox,int aCoul) :
    mW    (aW),
    mBox  (aBox),
    mCoul (aCoul)
{
    // Efface();
}
bool cCaseX11Xml::Inside(const Pt2di & aP) const { return mBox.inside(aP); }


int  cCaseX11Xml::GetCase(const std::vector<cCaseX11Xml *> & aVC,const Pt2di & aP)
{
     for (int  aKC=0 ; aKC<int (aVC.size()) ; aKC++)
     {
          if (aVC[aKC]->Inside(aP))
             return aKC;
     }
     return -1;
}


Clik  cCaseX11Xml::clik_in()
{
   while (1)
   {
      Clik aClk = mW.clik_in();
      if (Inside(Pt2di(aClk._pt)))
         return aClk;
   }
   return mW.clik_in();
}

     // ============== cWindowXmlEditor =======================================


         //=================================================

cWXXInfoCase::cWXXInfoCase(cElXMLTree * aTree,cElXMLTree * aFilter) :
   mTree      (aTree  ),
   mFilter    (aFilter),
   mTimeModif  (-1  )
{
}

bool cWXXTreeSelector::SelectTree(cElXMLTree *) {return true;}

cWXXVinoSelector::cWXXVinoSelector(const std::string & aName) :
   mName (aName)
{
}

bool  cWXXVinoSelector::SelectTree(cElXMLTree * aTree) 
{
   if (aTree->ValTag() != "Stats") return true;

    cElXMLTree * aTrName = aTree->GetUnique("NameFile");

    return aTrName->GetUniqueVal() == mName;
}

         //=================================================

cWindowXmlEditor::cWindowXmlEditor
(
     Video_Win aW,
     bool      aXmlMode,
     cElXMLTree * aTree,
     cWXXTreeSelector * aSelector,
     cElXMLTree * aFilter 
) :
    mFirstDraw    (true),
    mW            (aW),
    mXmlMode      (aXmlMode),
    mTreeGlob     (aTree),
    mSelector     (aSelector),
    mFilterGlob   (aFilter),
    mPRab         (Pt2di(12,12)),
    mCaseQuit     (0),
    mCaseWarn     (0),
    mGrayFond     (mXmlMode ?196 : 128),
    mGrayTag      (mXmlMode ?230 : 230),
    mSpaceTag     (mXmlMode?7:0),
    mDecalX       (mXmlMode? 30 : 0),
    mTimeModif   (0)
{
}


extern bool GenereErrorOnXmlInit;
extern bool GotErrorOnXmlInit;

void cWindowXmlEditor::ModifyCase(cCaseX11Xml * aCase,int aKC)
{

   aCase->Efface(P8COL::green);
   cWXXInfoCase & anIC = mVInfoCase[aKC];
   std::string & aStr0 = anIC.mTree->GetUniqueVal();
   std::string  aStrInit = aStr0;

   aStr0 = mW.GetString(Pt2dr(aCase->P0Line()),mW.pdisc()(P8COL::black),mW.pdisc()(P8COL::red),aStr0);
   bool Ok = true;
   cElXMLTree * aFilter = anIC.mFilter;


   {
       if ((aFilter!=0) &&  aFilter->HasAttr("Type"))
       {
           bool CurrErrorOnXmlInit = GenereErrorOnXmlInit;
           GenereErrorOnXmlInit = false;
           const std::string & aType = aFilter->ValAttr("Type");
           // std::cout << "TYPEEEE " << aType << "\n";
           if (aType=="int")
           {
                int anInt;
                xml_init(anInt,anIC.mTree);
           }
           else if (aType=="double")
           {
                double aDouble;
                xml_init(aDouble,anIC.mTree);
           }
           else if (aType=="Pt2dr")
           {
                Pt2dr aPt;
                xml_init(aPt,anIC.mTree);
           }
           else if (aType=="Pt2di")
           {
                Pt2di aPt;
                xml_init(aPt,anIC.mTree);
           }
           else
           {
                std::cout << "For type = " << aType << "\n";
                ELISE_ASSERT(false,"Bad Type Specif in filter Xml/X11");
           }

           GenereErrorOnXmlInit = CurrErrorOnXmlInit;
           if (GotErrorOnXmlInit)
           {
               ShowWarn(std::string("String=[") +aStr0 +  "] not valid for " +  aType , "Clik in to continue"); 


               // std::cout << "String=[" << aStr0 << "] not valid for " << aType << "\n";
               Ok = false;
               aStr0 = aStrInit;
           }
       }
   }

 
   aCase->Efface();
   aCase->string(-10,aStr0);
   if (Ok)
      anIC.mTimeModif = mTimeModif++;
}

void cWindowXmlEditor::Interact()
{
    while (1)
    {
        Clik aClik = mW.clik_in();
        int aKC =  cCaseX11Xml::GetCase(mVCase,round_ni(aClik._pt));
        if (aKC>=0)
        {
           cCaseX11Xml * aCase = mVCase[aKC];
           if (aCase == mCaseQuit)
           {
              return;
           }
           ModifyCase(aCase,aKC);
        }
    }
}


Box2di  cWindowXmlEditor::PrintTag(Pt2di aP0,cElXMLTree * aTree,int aMode,int aLevel,cElXMLTree * aFilter) 
{
    if ((!mXmlMode) && (aMode!=0)) 
       return Box2di(aP0-Pt2di(0,1),aP0-Pt2di(0,1));


    std::string aTag =  ((aMode == -1) ? "</" : "<") + aTree->ValTag() + ((aMode==0) ? "/>" : ">");
    if (!mXmlMode) aTag = " " + aTree->ValTag() + " :  ";
    Pt2di aSz = mW.SizeFixedString(aTag);

    Pt2di aP1 = aP0 + aSz;

    mW.fill_rect(Pt2dr(aP0)-Pt2dr(1,1),Pt2dr(aP1)+Pt2dr(1,1),mW.pgray()(mGrayTag));
    if (aMode!=0)
        mW.draw_rect(Pt2dr(aP0)-Pt2dr(2,2),Pt2dr(aP1)+Pt2dr(2,2),Line_St(mW.pdisc()(P8COL::blue),2));
    mW.fixed_string(Pt2dr(aP0)+Pt2dr(0,aSz.y), aTag.c_str(),mW.pdisc()(P8COL::black),false);

    Box2di aRes  (aP0-mPRab,aP1+ mPRab);
    if ((aMode ==0) && mFirstDraw)
    {
         Pt2di aQ0 (aP0.x+aSz.x+5,aP0.y-4);
         Pt2di aQ1 (EndXOfLevel(aLevel)-5,aP0.y+aSz.y+4);
         mVCase.push_back(cCaseX11Xml::Alloc(mW,Box2di(aQ0,aQ1),P8COL::yellow));
         mVInfoCase.push_back(cWXXInfoCase(aTree,aFilter));
    }

    return aRes;
}

int  cWindowXmlEditor::EndXOfLevel(int aLevel)
{
   return mW.sz().x-((aLevel+1)*mDecalX)/2;
}


Box2di cWindowXmlEditor::Draw(Pt2di aP0,cElXMLTree * aTree,int aLev,cElXMLTree * aFilter)
{
     Box2di aBoxNone(aP0,aP0);
     if (! mSelector->SelectTree(aTree)) 
        return aBoxNone;


     if (aTree->Profondeur() <= 1)
     {
          return PrintTag(aP0,aTree,0,aLev,aFilter);
     }

     aP0.y +=mSpaceTag;

     Box2di aRes = PrintTag(aP0,aTree,1,aLev,aFilter);

      for
      (
            std::list<cElXMLTree *>::iterator itF= aTree->Fils().begin();
            itF != aTree->Fils().end();
            itF++
      )
      {
           cElXMLTree * aFilsFilter = aFilter ? aFilter->GetOneOrZero((*itF)->ValTag()) : 0;

           if ((aFilter==0)  || (aFilsFilter!=0))
           {
              Box2di aBox = Draw(Pt2di(aP0.x+mDecalX,aRes._p1.y),*itF,aLev+1,aFilsFilter);
              aRes = Sup(aRes,aBox);
           }
      }

      
     Box2di aBoxFerm = PrintTag(Pt2di(aP0.x,aRes._p1.y),aTree,-1,aLev,aFilter);

     aRes =  Sup(aBoxFerm,aRes);
     if (mXmlMode)
     {
         mW.draw_rect(Pt2dr(aRes._p0),Pt2dr(EndXOfLevel(aLev),aRes._p1.y),mW.pdisc()(P8COL::red));
     }
     aRes._p1.y += mSpaceTag; 

     return aRes;
}

cWXXInfoCase *  cWindowXmlEditor::GetCaseOfNam(const std::string & aName,bool SVP)
{
   for (int aKC=0 ; aKC<int(mVInfoCase.size()) ; aKC++)
   {
        if (mVInfoCase[aKC].mTree && mVInfoCase[aKC].mTree->ValTag() == aName)
           return &(mVInfoCase[aKC]);
   }
   if (! SVP)
   {
       ELISE_ASSERT(false,"cWindowXmlEditor::GetCaseOfNam");
   }
   return  0;
}


void  cWindowXmlEditor::ShowQuit()
{
    mCaseQuit->Efface();
    mCaseQuit->string(0,"Quit edit");
}

void cWindowXmlEditor::ShowWarn(const std::string& aMes1, const std::string& aMes2)
{
   mCaseWarn->Efface(P8COL::red);

   mW.fixed_string_middle(Box2di(Pt2di(0,5),Pt2di(mW.sz().x,20)),0,aMes1,mW.pdisc()(P8COL::black),true);
   mW.fixed_string_middle(Box2di(Pt2di(0,25),Pt2di(mW.sz().x,40)),0,aMes2,mW.pdisc()(P8COL::black),true);


   mCaseWarn->clik_in();
   mCaseWarn->Efface(mW.pgray()(mGrayFond));

   ShowQuit();
}

Box2di cWindowXmlEditor::TopDraw()
{
    ELISE_COPY(mW.all_pts(),mGrayFond,mW.ogray());

    int aTx = mW.sz().x;
    // int aXMil = aTx/2;

    if (mFirstDraw)
    {
       mCaseQuit = cCaseX11Xml::Alloc(mW,Box2di(Pt2di(10,10),Pt2di(200,40)),P8COL::magenta);
       mCaseWarn = cCaseX11Xml::Alloc(mW,Box2di(Pt2di(10,2),Pt2di(aTx-2,48)),P8COL::red);
       mVCase.push_back(mCaseQuit);
       mVInfoCase.push_back(cWXXInfoCase(0,0));
    }

    Box2di aRes =  Draw(Pt2di(50,50),mTreeGlob,0,mFilterGlob);
    mFirstDraw = false;
    for (int aKC=0 ; aKC<int(mVCase.size()) ; aKC++)
    {
        cCaseX11Xml * aCX   = mVCase[aKC];
        aCX->Efface();
        cWXXInfoCase & anIC = mVInfoCase[aKC];
        cElXMLTree * aTree = anIC.mTree;
        if (aTree)
           aCX->string(-10,anIC.mTree->GetUniqueVal());
    }

    ShowQuit();
    return aRes;
   
}


void TestXmlX11()
{
    Video_Win aW =  Video_Win::WStd(Pt2di(700,800),1.0);
     
    cElXMLTree aFullTreeParam("EnvVino.xml");
    cWXXVinoSelector aSelector("FXDiv10.tif");
    cElXMLTree aFilter("FilterVino.xml");

    cWindowXmlEditor aWX(aW,true,aFullTreeParam.Fils().front(),&aSelector,&aFilter);

    // cWXXTreeSelector aSelId;
  
    aWX.TopDraw();
    aWX.Interact();

    aW.clear(); aW.clik_in();


    aWX.TopDraw();
    aWX.Interact();

    aFullTreeParam.StdShow("SORTIE.xml");

}


#endif





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
