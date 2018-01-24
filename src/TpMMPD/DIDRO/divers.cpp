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
    void changeImRadiom(std::vector<std::string> aLIm); //list image

    std::string mDir;
    private:
    std::string mFullDir;
    std::string mPat;
    std::string mHomog;
    std::string mOri;
    std::string mPrefixReech;
    bool mOverwrite;
    Pt2di mImSzOut;// si je veux découper mes images output, ex: homography between 2 sensors of different shape and size (TIR 2 VIS) but I want to have the same dimension as output
    Pt2di mRadiomRange;// If I want to change radiometry value, mainly to convert 16 bits to 8 bits
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
                    << EAM(mRadiomRange,"RadiomRange",true, "range of radiometry of input images, if given, output will be 8 bits images")
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

    // change the image radiometry
    if (EAMIsInit(&mRadiomRange))
    {
        std::cout << "Change images dynamic from range " << mRadiomRange << " to [0, 255] \n";
        changeImRadiom(aSetIm);
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

                bool Exist= ELISE_fp::exist_file(T2Reech_imName(_SetIm.at(aK)));

                if(!Exist || mOverwrite) {

                    std::cout << "aCom = " << aCom << std::endl;
                    //system_call(aCom.c_str());
                    aLCom.push_back(aCom);
                } else {
                    std::cout << "Reech image " << T2Reech_imName(_SetIm.at(aK)) << " exist, use F=1 to overwrite \n";
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
    // y sauver l'image
    ELISE_COPY(mTifIn.all_pts(),mTifIn.in(),im.out());
    // juste clipper
    Tiff_Im  aTifOut
             (
                 T2Reech_imName(imTIR).c_str(),
                 im.sz(),
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    // on écrase le fichier tif
   ELISE_COPY(im.all_pts(),im.in(),aTifOut.out());
    }
}

void cTIR2VIS_Appli::changeImRadiom(std::vector<std::string> aLIm)
{
    for(auto & imTIR: aLIm)
    {

    int minRad(mRadiomRange.x), rangeRad(mRadiomRange.y-mRadiomRange.x);

    // load reech images
    Tiff_Im mTifIn=Tiff_Im::StdConvGen(T2Reech_imName(imTIR),1,true);
    // create empty RAM image for imput image
    Im2D_REAL4 imIn(mTifIn.sz().x,mTifIn.sz().y);
    // create empty RAM image for output image
    Im2D_U_INT1 imOut(mTifIn.sz().x,mTifIn.sz().y);
    // fill it with tiff image value
    ELISE_COPY(
                mTifIn.all_pts(),
                mTifIn.in(),
                imIn.out()
               );

    // change radiometry
    for (int v(0); v<imIn.sz().y;v++)
    {
        for (int u(0); u<imIn.sz().x;u++)
        {
            Pt2di pt(u,v);
            double aVal = imIn.GetR(pt);
            unsigned int v(0);

            if(aVal!=0){
            if (aVal>minRad && aVal <minRad+rangeRad)
            {
                v=255.0*(aVal-minRad)/rangeRad;
            }
            }

            imOut.SetR(pt,v);
            //std::cout << "aVal a la position " << pt << " vaut " << aVal << ", transfo en " << v <<"\n";
        }
    }

    // remove file to be sure of result
    //ELISE_fp::RmFile(T2Reech_imName(imTIR));

    Tiff_Im aTifOut
             (
                 T2Reech_imName(imTIR).c_str(),
                 imOut.sz(),
                 GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    // on écrase le fichier tif
   ELISE_COPY(imOut.all_pts(),imOut.in(),aTifOut.out());
    }
}









/*    comparaise des orthos thermiques pour déterminer un éventuel facteur de calibration spectrale entre 2 frame successif, expliquer pouquoi tant de variabilité spectrale est présente (mosaique moche) */
// à priori ce n'est pas ça du tout, déjà mauvaise registration TIR --> vis du coup les ortho TIR ne se superposent pas , du coup correction radiometrique ne peut pas fonctionner.
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


// j'ai utilisé saisieAppui pour saisir des points homologues sur plusieurs couples d'images TIR VIS orienté
// je dois manipuler le résulat pour le tranformer en set de points homologues pour un unique couple d'images
// de plus, la saisie sur les im TIR est effectué sur des images rééchantillonnées, il faut appliquer une homographie inverse au points saisi
int TransfoMesureAppuisVario2TP_main(int argc,char ** argv)
{
    std::string a2DMesFileName, aOutputFile1, aOutputFile2,aImName("AK100419.tif"), aNameMap, aDirHomol("Homol-Man");

    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(a2DMesFileName, "Input mes2D file",  eSAM_IsExistFile)
                << EAMC(aNameMap, "Input homography to apply to TIR images measurements",  eSAM_IsExistFile),
    LArgMain()  << EAM(aImName,"ImName", true, "Name of Image for output files",  eSAM_IsOutputFile)
                << EAM(aOutputFile1,"Out1", true,  "Output TP file 1, def Homol-Man/PastisTIR_ImName/VIS_ImName.txt",  eSAM_IsOutputFile)
                << EAM(aOutputFile2,"Out2", true,  "Output TP file 2, def Homol-Man/PastisVIS_ImName/TIR_ImName.txt",  eSAM_IsOutputFile)
    );

    if (!EAMIsInit(&aOutputFile1)) {
        aOutputFile1=aDirHomol + "/PastisTIR_" + aImName + "/VIS_" + aImName + ".txt";
        if(!ELISE_fp::IsDirectory(aDirHomol)) ELISE_fp::MkDir(aDirHomol);
        if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisTIR_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisTIR_" + aImName);
    }
    if (!EAMIsInit(&aOutputFile2)) {
        aOutputFile2=aDirHomol + "/PastisVIS_" + aImName + "/TIR_" + aImName + ".txt";
        if(!ELISE_fp::IsDirectory(aDirHomol)) ELISE_fp::MkDir(aDirHomol);
        if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisVIS_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisVIS_" + aImName);
    }

    // lecture de la map 2D
    cElMap2D * aMap = cElMap2D::FromFile(aNameMap);

    // conversion de la map 2D en homographie; map 2D: plus de paramètres que l'homographie

    //1) grille de pt sur le capteur thermique auquel on applique la map2D
    ElPackHomologue  aPackHomMap2Homogr;
    for (int y=0 ; y<720; y +=10)
        {
         for (int x=0 ; x<1200; x +=10)
            {
             Pt2dr aPt(x,y);
             Pt2dr aPt2 = (*aMap)(aPt);
             ElCplePtsHomologues Homol(aPt,aPt2);
             aPackHomMap2Homogr.Cple_Add(Homol);
            }
        }
    // convert Map2D to homography
    cElHomographie H(aPackHomMap2Homogr,true);
    //H = cElHomographie::RobustInit(qual,aPackHomImTer,bool Ok(1),1, 1.0,4);

    // initialiser le pack de points homologues
    ElPackHomologue  aPackHom;

    cSetOfMesureAppuisFlottants aSetOfMesureAppuisFlottants=StdGetFromPCP(a2DMesFileName,SetOfMesureAppuisFlottants);

    int count=0;

    for( std::list< cMesureAppuiFlottant1Im >::const_iterator iTmes1Im=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().begin();
         iTmes1Im!=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().end();          iTmes1Im++    )
    {
        cMesureAppuiFlottant1Im anImTIR=*iTmes1Im;

        //std::cout<<anImTIR.NameIm().substr(0,5)<<" \n";
        // pour chacune des images thermique rééchantillonnée, recherche l'image visible associée
        if (anImTIR.NameIm().substr(0,5)=="Reech")
        {
            //std::cout<<anImTIR.NameIm()<<" \n";


            for (auto anImVIS : aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im()) {
            // ne fonctionne que pour la convention de préfixe Reech_TIR_ et VIS_
               if(anImTIR.NameIm().substr(10,anImTIR.NameIm().size()) == anImVIS.NameIm().substr(4,anImVIS.NameIm().size()))
               {
                   // j'ai un couple d'image.
                   //std::cout << "Couple d'images " << anImTIR.NameIm() << " et " <<anImVIS.NameIm() << "\n";

                   for (auto & appuiTIR : anImTIR.OneMesureAF1I())
                   {
                   //
                       for (auto & appuiVIS : anImVIS.OneMesureAF1I())
                       {
                       if (appuiTIR.NamePt()==appuiVIS.NamePt())
                       {
                           // j'ai 2 mesures pour ce point
                          // std::cout << "Pt " << appuiTIR.NamePt() << ", " <<appuiTIR.PtIm() << " --> " << appuiVIS.PtIm() << "\n";

                           // J'ajoute ce point au set de points homol
                           ElCplePtsHomologues Homol(appuiTIR.PtIm(),appuiVIS.PtIm());

                           aPackHom.Cple_Add(Homol);

                           count++;
                           break;
                       }
                       }
                   }
                   break;
               }
            }
       }

    // fin iter sur les mesures appuis flottant
    }
    std::cout << "Total : " << count << " tie points read \n" ;

    if (!EAMIsInit(&aOutputFile1) && !EAMIsInit(&aOutputFile2))
    {
    if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisReech_TIR_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisReech_TIR_" + aImName);
    std::cout << "Homol pack saved in  : " << aDirHomol + "/PastisReech_TIR_" + aImName + "/VIS_" + aImName + ".txt" << " \n" ;
    aPackHom.StdPutInFile(aDirHomol + "/PastisReech_TIR_" + aImName + "/VIS_" + aImName + ".txt");
    aPackHom.SelfSwap();

    std::cout << "Homol pack saved in  : " << aDirHomol + "/PastisVIS_" + aImName + "/Reech_TIR_" + aImName + ".txt" << " \n" ;
    aPackHom.StdPutInFile(aDirHomol + "/PastisVIS_" + aImName + "/Reech_TIR_" + aImName + ".txt");

    }


    // appliquer l'homographie

    //aPackHom.ApplyHomographies(H.Inverse(),H.Id());
    aPackHom.ApplyHomographies(H,H.Id());
    // maintenant on sauve ce pack de points homologues
    std::cout << "Homol pack saved in  : " << aOutputFile1 << " \n" ;
    aPackHom.StdPutInFile(aOutputFile1);
    aPackHom.SelfSwap();
    std::cout << "Homol pack saved in  : " << aOutputFile2 << " \n" ;
    aPackHom.StdPutInFile(aOutputFile2);
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





int main_test(int argc,char ** argv)
{
     //cORT_Appli anAppli(argc,argv);
     //CmpOrthosTir_main(argc,argv);
    //ComputeStat_main(argc,argv);
    //RegTIRVIS_main(argc,argv);
    //test_main(argc,argv);
    //MasqTIR_main(argc,argv);
    //cERO_ModelOnePaire(argc,argv);
    TransfoMesureAppuisVario2TP_main(argc,argv);

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

