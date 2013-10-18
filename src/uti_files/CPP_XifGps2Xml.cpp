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
// #include "XML_GEN/all_tpl.h" 

class cAppli_XifGps2Xml ;
class cIm_XifGp;


class cIm_XifGp
{
     public :
         cIm_XifGp(const std::string & aName,cAppli_XifGps2Xml &);

         cAppli_XifGps2Xml * mAppli;
         std::string         mName;
         cMetaDataPhoto      mMDP;
         bool                mHasPT;
         Pt3dr               mPGeoC;
};


class cAppli_XifGps2Xml : public cAppliListIm
{
    public :
       cAppli_XifGps2Xml(const std::string & aFullName,double aDefZ);
       void ExportSys(cSysCoord *,const std::string & anOri);
       
    public :
      std::vector<cIm_XifGp>  mVIm;
      double                  mDefZ;
      double                  mNbOk;
      Pt3dr                   mGeoCOriRTL;
      Pt3dr                   mWGS84DegreeRTL;
      cSystemeCoord           mSysRTL;
      cOrientationConique     mOC0;
};


/*******************************************************************/
/*                                                                 */
/*               cAppliListIm                                      */
/*                                                                 */
/*******************************************************************/


cAppliListIm::cAppliListIm(const std::string & aFullName) :
    mFullName (aFullName)
{
#if (ELISE_windows)
        replace( mFullName.begin(), mFullName.end(), '\\', '/' );
#endif
   SplitDirAndFile(mDir,mPat,mFullName);
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetIm = mICNM->Get(mPat);
}

/*******************************************************************/
/*                                                                 */
/*               cAppli_XifGps2Xml                                 */
/*                                                                 */
/*******************************************************************/



cAppli_XifGps2Xml::cAppli_XifGps2Xml(const std::string & aFullName,double aDefZ) :
   cAppliListIm (aFullName),
   mDefZ        (aDefZ),
   mNbOk        (0)
{
    mOC0 = StdGetFromPCP(Basic_XML_MM_File("Template-OrCamAng.xml"),OrientationConique);
    Pt3dr aPMoy(0,0,0);
    for (int aKI=0 ; aKI<int(mSetIm->size()) ; aKI++)
    {
          mVIm.push_back(cIm_XifGp((*mSetIm)[aKI],*this));
          if (mVIm.back().mHasPT)
          {
              aPMoy = aPMoy + mVIm.back().mPGeoC;
              mNbOk ++;
          }
    }
    if (mNbOk)
    {
        mGeoCOriRTL = aPMoy / double(mNbOk);
        mWGS84DegreeRTL = cSysCoord::WGS84Degre()->FromGeoC(mGeoCOriRTL);
        mSysRTL = StdGetFromPCP(Basic_XML_MM_File("Pattron-SysCoRTL.xml"),SystemeCoord);

        mSysRTL.BSC()[0].AuxR()[0] = mWGS84DegreeRTL.x;
        mSysRTL.BSC()[0].AuxR()[1] = mWGS84DegreeRTL.y;
        mSysRTL.BSC()[0].AuxR()[2] = mWGS84DegreeRTL.z;
    }
}

void cAppli_XifGps2Xml::ExportSys(cSysCoord * aSC,const std::string & anOri)
{
    std::string aKeyExport = "NKS-Assoc-Im2Orient@-" + anOri;
    for (int aKI=0 ; aKI<int(mVIm.size()) ; aKI++)
    {
        const cIm_XifGp & anIm = mVIm[aKI];
        if (anIm.mHasPT)
        {
            cOrientationConique   anOC =  mOC0;
            anOC.Externe().Centre() = aSC->FromGeoC(anIm.mPGeoC);

            MakeFileXML(anOC,mDir+mICNM->Assoc1To1(aKeyExport,anIm.mName,true));
        }
    }
}

/*******************************************************************/
/*                                                                 */
/*                  cIm_XifGp                                      */
/*                                                                 */
/*******************************************************************/

cIm_XifGp::cIm_XifGp(const std::string & aName,cAppli_XifGps2Xml & anAppli) :
    mAppli (&anAppli),
    mName (aName),
    mMDP  (cMetaDataPhoto::CreateExiv2(mAppli->mDir+aName)),
    mHasPT (false)
{
   if (mMDP.HasGPSLatLon())
   {
       Pt3dr aPWGS84;
       aPWGS84.x = mMDP.GPSLon();
       aPWGS84.y = mMDP.GPSLat();
       aPWGS84.z = mMDP.HasGPSAlt() ? mMDP.GPSAlt() : 0.0;
       mHasPT = true;
       mPGeoC  = cSysCoord::WGS84()->ToGeoC(aPWGS84);

       std::cout << aName << " : WGS84(rad) " << aPWGS84 << " GeoC  " << mPGeoC << "\n";
   }
}


/*******************************************************************/
/*                                                                 */
/*                      ::                                         */
/*                                                                 */
/*******************************************************************/




int XifGps2Xml_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);
    std::string aFullName;
    std::string  anOri;
    bool  DoRTL = true;
    std::string  aNameRTL = "RTLFromExif.xml";
    std::string  aNameSys = aNameRTL;
    double aDefZ=0;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullName,"Full Name") 
                      << EAMC(anOri,"Orientation"),
           LArgMain() << EAM(DoRTL,"DoRTL",true,"Do Local Tangent RTL")
                      << EAM(aNameRTL,"RTL","Name RTL")
                      << EAM(aNameSys,"SysCo","System of coordinates, by default RTL created (RTLFromExif.xml)")
                      << EAM(aDefZ,"DefZ","Default value for altitude (def 0)")
    );


    cAppli_XifGps2Xml anAppli(aFullName,aDefZ);

    if (DoRTL)
    {
       ELISE_ASSERT(anAppli.mNbOk!=0,"No GPS data to compute RTL reference system");
       MakeFileXML(anAppli.mSysRTL,anAppli.mDir+aNameRTL);
    }

    cSysCoord * aSysCo = cSysCoord::FromFile(anAppli.mDir + aNameSys);
    anAppli.ExportSys(aSysCo,anOri);


    return 0;
}






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
