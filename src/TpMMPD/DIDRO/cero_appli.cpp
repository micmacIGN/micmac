#include "cero_appli.h"

cERO_Appli::cERO_Appli(int argc, char** argv)
{
    mDebug=false;
    mMinOverX_Y=60; // recouvrement minimum entre 2 images, calculé séparément pour axe y et x.
    mMinOverX_Y_fichierCouple=20; // si le couple est renseigné dans un fichier, on est moins exigent
    mPropPixRec=33 ; // recouvrement minimum effectif (no data enlevée) en proportion de pixels, 1/mPropPixRec
    mDirOut="EROS/" ;
    mDir="./";
    mFullName=".*.tif";
    mSaveSingleOrtho=false;
    mFileOutModels="RadiomEgalModels.xml";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName, "Pattern of orthophoto",eSAM_IsPatFile),
        LArgMain()  << EAM(mFileClpIm,"FileCpl",true,"File of images couples like the one determined by GrapHom/oriconvert for Tapioca File, prefix Ort_ not include")
                    << EAM(mFileOutModels,"Out",true,"xml file name for the computed radiometric equalization models, default 'RadiomEgalModels.xml'")
                    << EAM(mDebug,"Debug",true,"Print Messages and write intermediates results")
                    << EAM(mSaveSingleOrtho,"ExportSO",true,"Export Single ortho corrected, def false")
                    << EAM(mDirOut,"Dir",true,"Directory where to store all intermediate results, default 'EROS/'. If Debug==0, this directory is purged at the end of the process.")
    );
    // to do: corriger mDirOut si pas de "/" à la fin

    SplitDirAndFile(mDir,mPatOrt,mFullName);
    MakeFileDirCompl(mDirOut);
    mDirOut=mDir+mDirOut;
    // define the "working directory" of this session
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = mICNM->StdGetListOfFile(mPatOrt);

    // suppression du directory si il existe, en cas de relance de la commande plusieurs fois on nettoie les résultats précédents
    if(ELISE_fp::IsDirectory(mDirOut))
    {
       std::cout << "Purge of directory " << mDirOut << "\n";
       ELISE_fp::PurgeDirGen(mDirOut,1);
    }

    // on charge les orto

    for (auto & imName : mLFile){
        //read Ortho
        cImGeo aIm(mDir+imName);
        mLIm.push_back(aIm);
    }
    std::cout << mLFile.size() << " Orthoimages loaded.\n";

    // COUPLES d'ORTHO
    if (!ELISE_fp::exist_file(mFileClpIm) && EAMIsInit(&mFileClpIm))
    {
        std::cout << "Cannot load the file of image pairs " << mFileClpIm << "\n";
    }

    // soit on ouvre le fichier de couples, soit on calcule les couples
    if (ELISE_fp::exist_file(mFileClpIm) && EAMIsInit(&mFileClpIm))
    {
        // charge le fichier de couples et controle que les images soient biens chargées également et avec un minimun d'overlap
        std::cout << "Load Ortho couples file and inspect couples \n";
        loadImPair();

    } else {
        // détermine les couples en chargeant les orthos et en regardant si elles se recouvrent.
        std::cout << "Determine images couples \n";
        computeImCplOverlap();
    }

    // CALCUL un modèle d'égalisation pour chacun des couples
    computeModel4EveryPairs();

    // moyenne les différents modèles pour chacunes des images
    moyenneModelPerOrt();

    saveModelsGlob();

    // applique le modèle à chacune des images
    if (mSaveSingleOrtho) applyRE();

    // suppression du directory intermediate results
    if(!mSaveSingleOrtho &&  !mDebug && ELISE_fp::IsDirectory(mDirOut))
    {
       std::cout << "Removal of directory " << mDirOut << "\n";
       ELISE_fp::PurgeDir(mDirOut,1);
    }
}

void cERO_Appli::applyRE()
{

    /* if (ELISE_fp::exist_file(mDir+"MTDMaskOrtho.xml"))
        { System("cp " + mDir+"MTDMaskOrtho.xml"+ " " + mDirOut+"MTDMaskOrtho.xml"); } else { std::cout << "unable to copy file " << mDir+"MTDMaskOrtho.xml" <<"\n";}
        if (ELISE_fp::exist_file(mDir+"MTDOrtho.xml"))
        { System("cp " + mDir+"MTDOrtho.xml"+ " " + mDirOut+"MTDOrtho.xml"); } else { std::cout << "unable to copy file " << mDir+"MTDOrtho.xml" <<"\n";}
    */

    int it(0);
    for (auto & ortho : mLIm)
    {
        // meme nom que l'ortho mais dans un sous dossier
        std::string filename(mDirOut+ortho.Name());
        std::cout << "Apply radiometric egalization on ortho " << ortho.Name() << ", save result in directory " << mDirOut <<"\n";
        Im2D_REAL4 aIm(ortho.applyRE(mL2Dmod.at(it)));

            // temporary, but now i copy hidden part, incidence images and xml in order to be able to run Tawny with corrected images
            /*
            std::vector<std::string> Vfile;
            Vfile.push_back("PC" + ortho.Name().substr(3));
            Vfile.push_back("PC" + ortho.Name().substr(3, ortho.Name().size()-6) + "xml");
            Vfile.push_back("MTD-" + ortho.Name().substr(4)+ ".xml");
            Vfile.push_back("Incid" + ortho.Name().substr(3));
            for (auto & file : Vfile)
            { if (ELISE_fp::exist_file(mDir+file))
                { System("cp " + mDir+file+ " " + mDirOut+file); } else { std::cout << "unable to copy file " << mDir+file <<"\n";}
            }
            */


        // très peu approprié de mettre ça ici, mais mon jeu test sont les images thermiques de variocam
        /*
        Im2D_U_INT1 out(aIm.sz().x,aIm.sz().y);
        int minRad(27540), rangeRad(2546.0);
        ELISE_COPY
        (
        select(aIm.all_pts(), aIm.in()>minRad && aIm.in()<minRad+rangeRad && aIm.in()!=0),
        255*(aIm.in()-minRad)/rangeRad,
        out.out()
        );
        // je sais pas pourquoi j'ai besoin de ça mais si non, effet de bord
        ELISE_COPY
        (
        select(aIm.all_pts(), aIm.in()==0),
        0,
        out.out()
        );
        // ne pas oublier de changer le format d'écriture en u_int_4
*/
        // je sauve le résultats
        ELISE_COPY(
                    aIm.all_pts(),
                    aIm.in(),
                    Tiff_Im(filename.c_str(),
                            aIm.sz(),
                            GenIm::real4,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero).out()
                    );
        it++;
    }
}


void cERO_Appli::moyenneModelPerOrt()
{
    for (auto & ortName : mLFile)
    {
       // charge les modèles pour cette orthos et calcule la droite "moyenne"
       std::cout << "Compute one linear model of egalization for ortho " << ortName << "\n";
       loadEROSmodel4OneIm(ortName);
    }
}

void cERO_Appli::loadImPair()
{
    cSauvegardeNamedRel aSNRtmp=StdGetFromPCP(mFileClpIm,SauvegardeNamedRel);

    // ajout d'un prefix :
    for
            (auto & cple : aSNRtmp.Cple() )
    {
        // vérifie qu'il est besoin d'un préfix
        if (cple.N1().substr(0,3)!="Ort_")  cple = cple.AddPrePost("Ort_","");
        // vérifie que les orthos sont bien renseignées dans le pattern d'entrée, et vérification d'un overlap effectif entre orthos
        cImGeo * pt1(0), * pt2(0);

        for (auto & im: mLIm) {
            if (im.Name()==cple.N1()) pt1=&im;
            if (im.Name()==cple.N2()) pt2=&im;
        }

        // test
        if (pt1!=NULL && pt2!=NULL)
        {
            // au minimum 5% de recouvrement
            if (pt1->overlap(pt2,mMinOverX_Y_fichierCouple))
            {
                // on vérifier que le couple n'est pas déjà dans la liste des couples
                bool doublon=false;

                for (auto &cpleOK: mSNR.Cple()) if (cpleOK==cple || cple==cCpleString(cpleOK.N2(),cpleOK.N1())) doublon=true;

                if(!doublon)
                {
                    mSNR.Cple().push_back(cple);
                    if (mDebug) std::cout << "Add images pair " << cple.N1() << "-->"<< cple.N2() << "\n";
                }
            }else {
                std::cout << "For the orthos couples " << cple.N1() << " and " << cple.N2() << " , not enough overlap  \n";}

        } else { std::cout << "For the orthos couples " << cple.N1() << " and " << cple.N2() << " , at least one ortho not in the loaded images\n";}
    }
     std::cout << "Number of valid orthos pairs :" << mSNR.Cple().size() << ".\n";
}

// A faire: limiter les paires qui sont symétrique ,Im1 Im2 , baquer Im2 Im1
void cERO_Appli::computeImCplOverlap()
{
    int count(0);
    int it1(0);
    for (auto & im1 : mLIm)
    {
        int it2(0);
        for (auto & im2 : mLIm)
        {  
            if(it2>it1)
            {
            if(im1.Name()!=im2.Name() && im1.containTer(im2.center()) && im1.overlap(&im2,mMinOverX_Y))
            {
                // vérif ici que les images ont bien un recouvrement effectif suffisant, en enlevant les no data
                int pixCommun(im1.pixCommun(&im2));
                if (pixCommun> (im1.nbPix()/mPropPixRec) || pixCommun>10000)
                {
                        // ajout du couple
                        mSNR.Cple().push_back(cCpleString(im1.Name(),im2.Name()));
                        if (mDebug) std::cout << "Add images pair " << im1.Name() << "-->"<< im2.Name() << "\n";
                        count++;
               // } else {
                 //   if(mDebug) std::cout << "for images pair " << im1.Name() << "-->"<< im2.Name() << ", not enough overlap\n";

                }
            }
            }
            it2++;
        }
        it1++;
    }
    std::cout << "Number of orthos pairs :" << count << ".\n";
}


void cERO_Appli::computeModel4EveryPairs()
{
    std::list<std::string> aLCom;
    for (auto & cple : mSNR.Cple())
    {
         std::string aCom =    MMBinFile(MM3DStr) + " TestLib Ero "
                                + mDir + cple.N1()
                                + "  "
                                + mDir +cple.N2()
                                + " Debug=" + ToString(mDebug)
                                + " Dir=" + mDirOut
                                + " W1=1 W2=1 WIncid=1" // poid des images, test avec poid egal toutes les images
                                ;
         aLCom.push_back(aCom);
         if (mDebug) std::cout << aCom << "\n";
    }
    cEl_GPAO::DoComInParal(aLCom);
}

// charger les modeles et calculer mod moyen
void cERO_Appli::loadEROSmodel4OneIm(std::string aNameOrt)
{

    std::vector<Pt2dr> aCpleRad;
    std::vector<double> aVPond;

    std::string file(mDirOut+aNameOrt.substr(0, aNameOrt.size()-3) + "txt");

    ifstream aFichier(file.c_str());

    if(aFichier)
    {
        std::string aLine;

        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');

            if(aLine.size() != 0)
            {
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aVal1Str = strtok(aBuffer," ");
                std::string aVal2Str = strtok( NULL, " " );
                std::string aPoidStr = strtok( NULL, " " );
                double aVal1,aVal2,aPoid;
                FromString(aVal1,aVal1Str);
                FromString(aVal2,aVal2Str);
                FromString(aPoid,aPoidStr);
                //std::cout << "aPoid = " << aPoid << std::endl;
                aCpleRad.push_back(Pt2dr(aVal1,aVal2));
                aVPond.push_back(aPoid);
            }
        }
    aFichier.close();

    // LSQ matching
    cLSQ_2dline LSQ_mod(&aCpleRad,&aVPond);
    LSQ_mod.adjustModelL2();
    if (mDebug) LSQ_mod.affiche();

    // sauve le modèle dans la liste des modèles
    mL2Dmod.push_back(LSQ_mod.getModel());
    // exporte le modèle pour utilisation ultérieure
    }
    else
    {
        std::cout<< "Error While opening file" << file << '\n';
        // modèle "unité" pour cette image, pas de correction radiometrique donc - solution temporaire
        mL2Dmod.push_back(c2DLineModel(0,1));
    }
}

void cERO_Appli::saveModelsGlob(){
    int it(0);
    cListOfRadiomEgalModel aLRE;
    for (auto & OrtName : mLFile)
    {
       cModLin mod;
       mod.NameIm()=OrtName;
       mod.a()=mL2Dmod.at(it).getA();
       mod.b()=mL2Dmod.at(it).getB();
       aLRE.ModLin().push_back(mod);
       it++;
    }
    MakeFileXML(aLRE,mDir+mFileOutModels);
}


int EgalRadioOrto_main(int argc,char ** argv)
{
   cERO_Appli(argc,argv);
   return EXIT_SUCCESS;
}
