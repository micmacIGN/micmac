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
#include "cimgeo.h"
#include "cero_modelonepaire.h"


extern int RegTIRVIS_main(int , char **);


//    Applique une homographie à l'ensemble des images thermiques pour les mettres dans la géométrie des images visibles prises simultanément

class cTIR2VIS_Appli;
class cTIR2VIS_Appli
{
    public:
    void ReechThermicIm(std::vector<std::string> aPatImgs, std::string aHomog);
    void CopyOriVis(std::vector<std::string> aPatImgs, std::string aOri);
    cTIR2VIS_Appli(int argc,char ** argv);
    string T2V_imName(string tirName);
    string T2Reech_imName(string tirName);
    void changeImSize(std::vector<std::string> aLIm); //list image

    std::string mDir;
    private:
    std::string mFullDir;
    std::string mPat;
    std::string mHomog;
    std::string mOri;
    std::string mPrefixReech;
    bool mOverwrite;
    Pt2di mImSzOut;// si je veux découper mes images output, ex: homography between 2 sensors of different shape and size (TIR 2 VIS) but I want to have the same dimension as output

};


cTIR2VIS_Appli::cTIR2VIS_Appli(int argc,char ** argv) :
      mFullDir	("img.*.tif"),
      mHomog	("homography.xml"),
      mOri		("RTL"),
      mPrefixReech("Reech"),
      mOverwrite (false)



{
    ElInitArgMain
    (
    argc,argv,
        LArgMain()  << EAMC(mFullDir,"image pattern", eSAM_IsPatFile)
                    << EAMC(mHomog,"homography XML file", eSAM_IsExistFile ),
        LArgMain()  << EAM(mOri,"Ori",true, "ori name of VIS images", eSAM_IsExistDirOri )
                    << EAM(mOverwrite,"F",true, "Overwrite previous resampled images, def false")
                    << EAM(mImSzOut,"ImSzOut",true, "Size of output images")
    );


    if (!MMVisualMode)
    {

    SplitDirAndFile(mDir,mPat,mFullDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(mPat));


    ReechThermicIm(aSetIm,mHomog);

     if (EAMIsInit(&mOri))
     {
         StdCorrecNameOrient(mOri,mDir);
         mOri="Ori-"+mOri+"/";
         std::cout << "Copy orientation file." << std::endl;
         CopyOriVis(aSetIm,mOri);
      }

    // changer la taille des images out
    if (EAMIsInit(&mImSzOut))
    {
        //open first reech image just to read the dimension in order to print a message
        Tiff_Im mTif=Tiff_Im::StdConvGen(T2Reech_imName(aSetIm.at(0)),1,true);
        std::cout << "Change size of output images from " << mTif.sz() << " to " << mImSzOut << "\n";

        changeImSize(aSetIm);
    }

    }
}



void cTIR2VIS_Appli::ReechThermicIm(
                                      std::vector<std::string> _SetIm,
                                      std::string aHomog
                                      )
{

     std::list<std::string>  aLCom;

    for(unsigned int aK=0; aK<_SetIm.size(); aK++)
    {
                string  aNameOut = "Reech_" + NameWithoutDir(StdPrefix(_SetIm.at(aK))) + ".tif";// le nom default donnée par ReechImMap

                std::string aCom = MMDir()
                            + std::string("bin/mm3d")
                            + std::string(" ")
                            + "ReechImMap"
                            + std::string(" ")
                            + _SetIm.at(aK)
                            + std::string(" ")
                            + aHomog;

                            if (EAMIsInit(&mPrefixReech)) {  aCom += " PrefixOut=" + T2Reech_imName(_SetIm.at(aK)) ; }

                            //+ " Win=[3,3]";// taille de fenetre pour le rééchantillonnage, par défaut 5x5

                bool Exist= ELISE_fp::exist_file(aNameOut);

                if(!Exist || mOverwrite) {

                    std::cout << "aCom = " << aCom << std::endl;
                    //system_call(aCom.c_str());
                    aLCom.push_back(aCom);
                }
    }
    cEl_GPAO::DoComInParal(aLCom);
}

// dupliquer l'orientation des images visibles de la variocam pour les images thermiques accociées
void cTIR2VIS_Appli::CopyOriVis(
                                      std::vector<std::string> _SetIm,
                                      std::string aOri
                                      )
{

    for(auto & imTIR: _SetIm)
    {
        //cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
        std::string aOriFileName(aOri+"Orientation-"+T2V_imName(imTIR)+".xml");
        if (ELISE_fp::exist_file(aOriFileName))
        {
        std::string aCom="cp " + aOriFileName + "   "+ aOri+"Orientation-" + T2Reech_imName(imTIR) +".xml";
        std::cout << "aCom = " << aCom << std::endl;
        system_call(aCom.c_str());
        } else
        {
        std::cout << "Can not copy orientation " << aOriFileName << " because file not found." << std::endl;
        }

    }
}



string cTIR2VIS_Appli::T2V_imName(string tirName)
{
   std::string visName=tirName;

   visName[0]='V';
   visName[2]='S';

   return visName;

}

string cTIR2VIS_Appli::T2Reech_imName(string tirName)
{
   return mPrefixReech+ "_" + tirName;
}



int T2V_main(int argc,char ** argv)
{
    cTIR2VIS_Appli aT2V(argc,argv);
    return EXIT_SUCCESS;
}


void cTIR2VIS_Appli::changeImSize(std::vector<std::string> aLIm)
{
    for(auto & imTIR: aLIm)
    {
    // load reech images
    Tiff_Im mTifIn=Tiff_Im::StdConvGen(T2Reech_imName(imTIR),1,true);
    // create RAM image
    Im2D_REAL4 im(mImSzOut.x,mImSzOut.y);
    // compute the translation for initial to final image size -- attention, que ce passe t il si l'offset en pixel n'est pas un nombre pair à diviser en 2?
    Pt2di Tr((mTifIn.sz().x-im.sz().x)/2, (mTifIn.sz().x-im.sz().x)/2);

    ELISE_COPY
   (
   mTifIn.all_pts(),
   trans(mTifIn.in(),Tr),
   im.out()
   );
    // on écrase le fichier tif
    Tiff_Im  aTifOut
             (
                 T2Reech_imName(imTIR).c_str(),
                 im.sz(),
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    }
}


/*    comparaise des orthos thermiques pour déterminer un éventuel facteur de calibration spectrale entre 2 frame successif, expliquer pouquoi tant de variabilité spectrale est présente (mosaique moche) */

int CmpOrthosTir_main(int argc,char ** argv)
{
    std::string aDir, aPat="Ort_.*.tif", aPrefix="ratio";
    int aScale = 1;
    bool Test=true;
    std::list<std::string> mLFile;
    std::vector<cImGeo> mLIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Ortho's Directory", eSAM_IsExistFile),
        LArgMain()  << EAM(aPat,"Pat",false,"Ortho's image pattern, def='Ort_.*'",eSAM_IsPatFile)
                    << EAM(aScale,"Scale",false,"Scale factor for both Orthoimages ; Def=1")
                    << EAM(Test,"T",false, "Test filtre des bords")
                    << EAM(aPrefix,"Prefix", false,"Prefix pour les ratio, default = ratio")

    );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);


    for (auto & imName : mLFile){
        //read Ortho

        cImGeo aIm(imName);
        std::cout << "nom image " << aIm.Name() << "\n";
        mLIm.push_back(aIm);
    }
    std::cout << mLFile.size() << " Ortho chargées.\n";

    //  tester si l'overlap est suffisant
    int i(0);
    for (auto aCurrentImGeo: mLIm)
    {
    i++;
    for (int j=i ; j<mLFile.size(); j++)
    {

        if (mLIm.at(j).Name()!=aCurrentImGeo.Name() && mLIm.at(j).overlap(&aCurrentImGeo,70))
        {

        Pt2di aTr=aCurrentImGeo.computeTrans(&mLIm.at(j));
        //std::string aName="ratio"+std::to_string(j)+"on"+std::to_string(j+1)+".tif";
        std::string aName=aPrefix+aCurrentImGeo.Name()+"on"+mLIm.at(j).Name()+".tif";
        // copie sur disque de l'image // pas très pertinent, je devrais plutot faire tout les calcul en ram puis sauver l'image à la fin avec un constructeur de cImGeo qui utilise une image RAM et les info du georef
        cImGeo        aImGeo(& aCurrentImGeo, aName);

        aImGeo.applyTrans(aTr);

        Im2D_REAL4 aIm=aImGeo.toRAM();
        Im2D_REAL4 aIm2(aIm.sz().x, aIm.sz().y);
        Im2D_REAL4 aImEmpty(aIm.sz().x, aIm.sz().y);
        Im2D_REAL4 aIm3=mLIm.at(j).toRAM();

        // l'image 1 n'as pas la meme taille, on la copie dans une image de meme dimension que l'im 0
        ELISE_COPY
                (
                    aIm3.all_pts(),
                    aIm3.in(),// l'image 1 n'as pas la meme taille, on la copie dans une image de meme dimension que l'im 0n(),
                    aIm2.oclip()
                    );

        // division de im 0 par im 1
        ELISE_COPY
                (
                    select(aIm.all_pts(),aIm2.in()>0),
                    (aIm.in())/(aIm2.in()),
                    aImEmpty.oclip()
                    );

        if (Test){
        // etape de dilation, effet de bord non désiré
        int it(0);
        do{

        Neighbourhood V8 = Neighbourhood::v8();
        Liste_Pts_INT2 l2(2);

        ELISE_COPY
        (
        dilate
        (
        select(aImEmpty.all_pts(),aImEmpty.in()==0),
        sel_func(V8,aImEmpty.in_proj()>0)
        ),
        1000,// je me fous de la valeur c'est pour créer un flux de points surtout
        aImEmpty.out() | l2 // il faut écrire et dans la liste de point, et dans l'image, sinon il va repecher plusieur fois le meme point
        );
        // j'enleve l'effet de bord , valleurs nulles
        ELISE_COPY
                (
                    l2.all_pts(),
                    0,
                    aImEmpty.out()
                    );

        it++;

        } while (it<3);

        }
        // je sauve mon image RAM dans mon image tif file
        aImGeo.updateTiffIm(&aImEmpty);

        // je calcule la moyenne du ratio
        int nbVal(0);
        double somme(0);
        for(int aI=0; aI<aImEmpty.sz().x; aI++)
        {
            for(int aJ=0; aJ<aImEmpty.sz().y; aJ++)
            {
                Pt2di aCoor(aI,aJ);
                double aValue = aImEmpty.GetR(aCoor);
                if (aValue!=0) {
                    somme +=aValue;
                    nbVal++;
                    //std::cout <<"Valeur:"<<aValue<< "\n";
                }
            }
            //fprintf(aFP,"\n");
        }
        somme/=nbVal;

        std::cout << "Ratio de l'image " << aCurrentImGeo.Name() << " sur l'image " << mLIm.at(j).Name() << "  caclulé, moyenne de  "<< somme << " ------------\n";
        // end if
        }
        // end boucle 1
    }
    // end boucle 2
    }
    return EXIT_SUCCESS;
}


int ComputeStat_main(int argc,char ** argv)
{
    double NoData(0);
    std::string aPat("Ree*.tif");
    std::list<std::string> mLFile;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aPat,"Image pattern",eSAM_IsPatFile),
                LArgMain()  << EAM(NoData,"ND", "no data value, default 0")
                );

    std::cout <<" Debut\n";
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);

    std::cout << mLFile.size() << " images\n";
    double aMax(0), aMin(0);

    for (auto & aName : mLFile){

    Tiff_Im mTif=Tiff_Im::StdConvGen(aName,1,true);
    Im2D_REAL4 aImRAM(mTif.sz().x, mTif.sz().y);
    ELISE_COPY
            (
                mTif.all_pts(),
                mTif.in(),
                aImRAM.out()
                );

    // je calcule la moyenne du ratio
    int nbVal(0);
    bool firstVal=1;
    double somme(0),min,max(0);
    for(int aI=0; aI<aImRAM.sz().x; aI++)
    {
        for(int aJ=0; aJ<aImRAM.sz().y; aJ++)
        {
            Pt2di aCoor(aI,aJ);
            double aValue = aImRAM.GetR(aCoor);
            if (aValue!=NoData) {
                if (firstVal)
                {
                    min=aValue;
                    firstVal=0;
                }
                if (aValue<min) min=aValue;
                if (aValue>max) max=aValue;
                somme +=aValue;
                nbVal++;
            }
        }
    }
    somme/=nbVal;
    std::cout <<"Statistique Image "<<aName<< "\n";
    std::cout << "Nb value !=" << NoData << " :" << nbVal << "\n";
    std::cout << "Mean :" << somme <<"\n";
    std::cout << "Max :" << max <<"\n";
    std::cout << "Min :" << min <<"\n";
    std::cout << "Dynamique (max-min) :" << max-min <<"\n";

    // stat sur toutes les images
    if (mLFile.front()==aName)
    {
        aMin=min;
        aMax=max;
    }

    if (max>aMax) aMax=max;
    if (min<aMin) aMin=min;

}
    std::cout << "Max de toutes les images :" << aMax <<"\n";
    std::cout << "Min de toutes les iamges :" << aMin <<"\n";
    std::cout << "Dynamique (max-min) :" << aMax-aMin <<"\n";

    return EXIT_SUCCESS;
}



int test_main(int argc,char ** argv)
{
    std::string aDir, aPat="Ort_.*.tif", aPrefix="box_";
    std::list<std::string> mLFile;
    std::vector<cImGeo> mLIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Ortho's Directory", eSAM_IsExistFile),
        LArgMain()  << EAM(aPat,"Pat",false,"Ortho's image pattern, def='Ort_.*'",eSAM_IsPatFile)
                    << EAM(aPrefix,"Prefix", false,"Prefix pour les ratio, default = box")
    );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);

    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
    {
    cImGeo aIm(*itS);

    aIm.display();
    std::string filename=aPrefix + aIm.Name();
    Pt2dr min(916320,6529220);
    Pt2dr max(916320+5,6529220+5);

    Im2D_REAL4 clipped=aIm.clipImTer(min,max);

    ELISE_COPY
    (
        clipped.all_pts(),
        clipped.in(),
        Tiff_Im(
            filename.c_str(),
            clipped.sz(),
            GenIm::real4,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );

    }
    return EXIT_SUCCESS;
}



int MasqTIR_main(int argc,char ** argv)
{
    std::string aDir, aPat="Ort_.*.tif";
    std::list<std::string> mLFile;
    std::vector<cImGeo> mLIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Ortho's Directory", eSAM_IsExistFile),
        LArgMain()  << EAM(aPat,"Pat",false,"Ortho's image pattern, def='Ort_.*'",eSAM_IsPatFile)

    );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);

    for (auto & aName : mLFile)
    {

    Tiff_Im im=Tiff_Im::StdConvGen("../OrthoTIR025/"+aName,1,true);

    std::string filenamePC= "PC"+aName.substr(3, aName.size()) ;

  /*
    //255: masqué. 0: ok
    Im2D_U_INT1 out(im.sz().x,im.sz().y);
    Im2D_REAL4 tmp(im.sz().x,im.sz().y);


    int minRad(27540), rangeRad(2546.0);


    ELISE_COPY
    (
    im.all_pts(),
    im.in(),
    tmp.out()
    );

    ELISE_COPY
    (
    select(tmp.all_pts(), tmp.in()>minRad && tmp.in()<minRad+rangeRad && tmp.in()!=0),
    255*(tmp.in()-minRad)/rangeRad,
    out.out()
    );

    ELISE_COPY
    (
    select(tmp.all_pts(), tmp.in()==0),
    0,
    out.out()
    );




    for (int v(0); v<tmp.sz().y;v++)
    {
        for (int u(0); u<tmp.sz().x;u++)
        {
            Pt2di pt(u,v);
            double aVal = tmp.GetR(pt);
            unsigned int v(0);

            if(aVal!=0){
            if (aVal>minRad && aVal <minRad+rangeRad)
            {
                v=255.0*(aVal-minRad)/rangeRad;
            }
            }

            out.SetR(pt,v);
            //std::cout << "aVal a la position " << pt << " vaut " << aVal << ", transfo en " << v <<"\n";
        }
    }






    std::cout << "je sauve l'image " << aName << "\n";
    ELISE_COPY
    (
        out.all_pts(),
        out.in(0),
        Tiff_Im(
            aName.c_str(),
            out.sz(),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );

*/
     Im2D_REAL4 masq(im.sz().x,im.sz().y);

    ELISE_COPY
    (
    im.all_pts(),
    im.in(0),
    masq.oclip()
    );


     std::cout << "détecte le bord pour image  " << filenamePC << "\n";
    int it(0);
    do{

    Neighbourhood V8 = Neighbourhood::v8();
    Liste_Pts_INT2 l2(2);

    ELISE_COPY
    (
    dilate
    (
    select(masq.all_pts(),masq.in()==0),
    sel_func(V8,masq.in_proj()>0)
    ),
    1000,// je me fous de la valeur c'est pour créer un flux de points surtout
    masq.oclip() | l2 // il faut écrire et dans la liste de point, et dans l'image, sinon il va repecher plusieur fois le meme point
    );
    // j'enleve l'effet de bord , valleurs nulles
    ELISE_COPY
            (
                l2.all_pts(),
                0,
                masq.oclip()
                );

    it++;

    } while (it<3);


/*
    // attention, écrase le ficher existant, pas propre ça
    std::cout << "je sauve l'image avec correction radiométrique " << aName << "\n";
    ELISE_COPY
    (
        masq.all_pts(),
        masq.in(0),
        Tiff_Im(
            aName.c_str(),
            masq.sz(),
            GenIm::int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );

*/
    ELISE_COPY
    (
    select(masq.all_pts(),masq.in()==0),
    255,
    masq.oclip()
    );

    ELISE_COPY
    (
    select(masq.all_pts(),masq.in()!=255),
    0,
    masq.oclip()
    );


    std::cout << "je sauve l'image les parties cachées " << filenamePC << "\n";
    ELISE_COPY
    (
        masq.all_pts(),
        masq.in(0),
        Tiff_Im(
            filenamePC.c_str(),
            masq.sz(),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );


    }
    return EXIT_SUCCESS;
}





int main_test2(int argc,char ** argv)
{
     //cORT_Appli anAppli(argc,argv);
     //CmpOrthosTir_main(argc,argv);
    //ComputeStat_main(argc,argv);
    RegTIRVIS_main(argc,argv);
    //test_main(argc,argv);
    //MasqTIR_main(argc,argv);
    //cERO_ModelOnePaire(argc,argv);


   return EXIT_SUCCESS;
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

A cet égard  l'attention de l'ucApplitilisateur est attirée sur les risques
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

