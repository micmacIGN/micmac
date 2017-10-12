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

//definir une structure qui contient (le coefficient, le temps, une liste des coordonnées X Y Z et l'indice du premier élement de la liste dans la liste principale)
struct aSolution
{
    std::vector<Pt3dr> Liste;
    double coefficient;
    int indice;
    int t;
};

//la declaration des variables et des fonction de la classe S_Appli
class S_Appli
{
public :
    S_Appli( int argc, char ** argv );

private :
    void ShowSommet();
    void ShowGPS();
    void ShowGPS_RTK();
    double distance(Pt3dr a,Pt3dr b);
    double calc_coef(std::vector<double> Ratio_S,std::vector<double> Ratio_G);
    double Somme(std::vector<double> Ratio);
    void coefficient();   
    void affiche_max(std::vector<aSolution> solution, int dt_max);
    void test_successiv();
    void sauvegarde();
    void test_dt(int best_dt);
    void test_exif();
    std::vector<double> calc_Ratios(std::vector<Pt3dr> aList);
    aSolution cherche_max(std::vector<aSolution> solution, int dt);
    std::vector<aSolution> Trie(int nbr_sol);
    std::string mDir;
    std::string mPat;
    std::string aOri ;
    std::string aGpsFile ;
    int nbr_sol;
    double nbr;
    std::list<std::string>files;
    std::list<std::string>aImgsFiles;
    std::string mFullName;
    cInterfChantierNameManipulateur *ManC;
    std::vector<Pt3dr> aSomList;
    std::vector<Pt3dr> aGPSList;
    std::vector<string> aNameList;
    std::vector<string> aNameImgs;
    std::vector<int> aDecalList;
    std::vector<aSolution> solution;
    std::vector<aSolution> all_Solutions;
    aSolution best;
    int duree;
    int dt_max;
    std::string aOut;
    std::string aDir;

};


S_Appli::S_Appli(int argc, char ** argv )
{
    nbr=0;
    
    ElInitArgMain
    (

        argc,argv,
        LArgMain()	<<EAMC(aDir,"Directory")
					<<EAMC(aOri,"Orientations directory", eSAM_IsExistDirOri)
					<<EAMC(aGpsFile,"Gps .xml file of trajectory", eSAM_IsExistFile)
					<<EAMC(mFullName,"Pattern of images", eSAM_IsPatFile),
        LArgMain()	<<EAM(nbr,"nbr",true,"Number of best solutions to display")
					<<EAM(aOut,"Out",true,"output file", eSAM_IsOutputFile)
    );
    
    
    
    //name output .xml file
    if (aOut=="")
    {
		aOut = "Ori-" + StdPrefixGen(aGpsFile) + ".txt";
    }

	SplitDirAndFile(mDir, mPat, mFullName);
    std::cout << "mDir =" << mDir << std::endl;
    std::cout << "mPat =" << mPat << std::endl;
    std::cout << "mFullName =" << mFullName << std::endl;
    
    ManC=cInterfChantierNameManipulateur::BasicAlloc(mDir);
    aImgsFiles=ManC->StdGetListOfFile(mPat);
   
    for(std::list<std::string>::iterator I=aImgsFiles.begin();I!=aImgsFiles.end();I++)
    {	
        //std::cout << "*I2 =" << *I << std::endl;
        aNameImgs.push_back(*I);
    }

    int nbr_i=0;
    nbr_i=int (nbr);
    
    if (nbr==nbr_i)   //tester si le nbr (si entré) par l'utilisateur est un entier
    {

        nbr_sol=nbr_i;        
        if(aGpsFile.compare(aGpsFile.size()-3,3,"xml") == 0)        	//test si l'extension du fichier GPS est .xml?
        {

            ShowGPS();		   //creation de la listes des points GPS à partir du fichier Xml en argument
        }
        
        else                    //test si l'extension du fichier GPS est .xml?
        {
            ShowGPS_RTK();      //creation de la listes des points GPS à partir du fichierRTKLib en argument
        }


         ShowSommet();      //creation de la listes des points Sommets à partir du dossier Ori en argument
							//(une liste des coord et une autre qui contient les noms


        test_successiv();   //test la successivité des images et prend en compte s'il ya un decalage entre les images


        coefficient();      //fournie deux vecteurs de type aSolution: 1/[all_Solution]:cherche toute les combinaisons possibles,leurs coef,l'indice de premier element ainsi que son dt
																	// 2/ [solution]:contient la combinaison optimale pour tout les dt possibles(le dt,indice,liste et le coefficient(max))

    if(nbr_sol<1)
    {
        affiche_max(solution ,  dt_max);        //affiche la solution optimale de chaque dt

    }

    else
    {
        std::vector<aSolution> combinaisons=Trie(nbr_sol);     //affiche les n premières solutions optimales dans un ordre décroissant (n est nbr_sol un nombre entier entré par l'utilisateur)
        affiche_max(combinaisons ,  nbr_sol);
    }


    sauvegarde();       //Parmis les meilleurs solutions trouvés de chaque dt dans la fonction coefficient(sauvegardés dans le vecteur solution),
                        //on sauvegarde la meilleur dans un fichier (resultat.txt)

    //test_exif();  //teste si les données exif existent sinon les génère
                    //et affiche si le dt entre les images est compatible avec le resultat trouvé sinon il affiche un warning
    }
    else
    {
        std::cout <<"NBR INCORRECTE " <<nbr<< '\n';
        std::cout <<"Entrer un nombre entier superieur à 1" << '\n';
    }
}





//ShowGPS() retourne aGPSListe (la liste des coordonnées des points GPS à partir du fichier xml)
void S_Appli::ShowGPS()  
{
    cDicoGpsFlottant aDico =  StdGetFromPCP(aGpsFile,DicoGpsFlottant);
    // std::list<cOneGpsDGF> &GPS = aDico.OneGpsDGF();
    // for(std::list<cOneGpsDGF>::iterator IT=GPS.begin();IT!=GPS.end();IT++)
    for(auto IT=aDico.OneGpsDGF().begin();IT!=aDico.OneGpsDGF().end();IT++)
    {
        aGPSList.push_back(IT->Pt());        
     }
}



//ShowGPS_RTK() retourne aGPSListe (la liste des coordonnées des points GPS à partir du fichier RTKLib)
void S_Appli::ShowGPS_RTK()  
{
    ifstream fichier(aGpsFile.c_str());  		//déclaration du flux et ouverture du fichier

            if(fichier)  						// si l'ouverture a réussi
            {

                std::string ligne; 				//Une variable pour stocker les lignes lues
                while(!fichier.eof())
                 {
                    getline(fichier,ligne);
                    std::cout<<ligne<<'\n';



                    if (ligne.compare(0,1,"%") == 0)		//pour sauter l'entête (toute les lignes qui commencent par "%")
                    {
                        std::cout<<"%%%%%%"<<'\n';
                    }
                    else if(ligne.size()>10)       //if(ligne.size()>10) --> pour résoudre le problème du dernière ligne du fihier
                    {
                      std::string s = ligne;
                      std::vector<string> coord;                 
                      int lowmark=-1;
                      int uppermark=-1;
                      for(unsigned int i=0;i<s.size()+1;i++)     // parser chaque ligne par l'espace
                      {
                          if(std::isspace(s[i]) && (lowmark!=-1))
                          {                             
                             string token = s.substr(lowmark,uppermark);                             
                             coord.push_back(token);
                             //nouveau mot
                             lowmark=-1;
                             uppermark=-1;
                          }
                          else
                              if(!(std::isspace(s[i])) && (lowmark==-1))
                          {
                              lowmark=i;
                              uppermark=i+1;
                          }
                          else
                          if(!(std::isspace(s[i])) && (lowmark!=-1))
                          {
                              uppermark++;
                          }else
                          {
                              lowmark=-1;
                              uppermark=-1;
                          }
                      }

                      Pt3dr pt;
                      double x = atof (coord[2].c_str());
                      pt.x=x;
                      std::cout<<"x "<<pt.x<<'\n';
                      double y = atof (coord[3].c_str());
                      pt.y=y;
                      std::cout<<"y "<<pt.y<<'\n';
                      double z = atof (coord[4].c_str());
                      pt.z=z;
                      std::cout<<"z "<<pt.z<<'\n';
                      aGPSList.push_back(pt);                      
                    }
                 }
                  fichier.close();  // on referme le fichier
            }
            else
                    std::cout<< "Erreur à l'ouverture !" << '\n';

}


//ShowSommet() retourne 2 listes: 'aSomList'(la liste des coordonnées des points sommets) et 'aNameListe'(la liste des noms des images)
void S_Appli::ShowSommet()       
{    
    string new_mFullName="Orientation-*.*xml";
    ManC=cInterfChantierNameManipulateur::BasicAlloc(aOri);
    files=ManC->StdGetListOfFile(new_mFullName);
    for(std::list<std::string>::iterator I=files.begin();I!=files.end();I++)
    {	
        //std::cout << "*I1 =" << *I << std::endl;
        aNameList.push_back(*I);
        cOrientationConique aOriConique=StdGetFromPCP(aOri+"/"+*I,OrientationConique);        
        aSomList.push_back(aOriConique.Externe().Centre());
    }
    
}



//test_successiv(): retourne	duree (la durée de prise des photos en unité de temps) 
				//				et aDecalList(une liste qui contient le decalage en unité de temps de chaque image avec la suivante)
void S_Appli::test_successiv()      
{
    duree=1;
    for (unsigned int i=0;i<aNameList.size()-1;i++)
    {
        string st1 = aNameList[i+1].substr(aNameList[i+1].find("2_0")+3, aNameList[i+1].find(".t"));
        stringstream ss(st1);
        int a;
        ss >> a;

        string st2 = aNameList[i].substr(aNameList[i].find("2_0")+2, aNameList[i].find(".t"));
        stringstream bb(st2);
        int b;
        bb >> b;
        int d=a-b;
        aDecalList.push_back(d);
        duree=duree+d;
    }
    aDecalList.push_back(1);

}


//la fonction distance calcule la distance entre 2 points
double S_Appli::distance(Pt3dr a,Pt3dr b)    
{
      double carr1x=((a.x-b.x)*(a.x-b.x));
      double carr1y=((a.y-b.y)*(a.y-b.y));
      double carr1z=((a.z-b.z)*(a.z-b.z));
      double dist1=sqrt(carr1x+carr1y+carr1z);
      return(dist1);
}

//la fonction calc_Ratios calcule les ratios de distances d'une liste de points
std::vector<double> S_Appli::calc_Ratios(std::vector<Pt3dr> aList)  
{
    std::vector<double> ratio;
    for (unsigned int i=0;i<aList.size()-1;i=i+2)
    {
      double dist1=distance(aList[i+1],aList[i]);
      double dist2=distance(aList[i+2],aList[i+1]);
      ratio.push_back(dist1/dist2);

    }
    return(ratio);

}


//la fonction somme calcule la somme de tous les élements d'une liste des ratios
double S_Appli::Somme(std::vector<double> Ratio)  
{
    double somme=0;
    for (unsigned int i=0;i<Ratio.size();i++)
    somme=somme+Ratio[i];
    return(somme);
}


//la fonction calc_coef calcule le coefficient de corrélation entre deux vecteurs
double S_Appli::calc_coef(std::vector<double> Ratio_S,std::vector<double> Ratio_G) 
{
    double Som_ratio_Som=Somme(Ratio_S);  //somme des ratios des points sommets
    double Som_ratio_GPS=Somme(Ratio_G);   //somme des ratios des points GPS
    double moy1=Som_ratio_Som/Ratio_S.size();
    double moy2=Som_ratio_GPS/Ratio_G.size();
    double coeff;
    double prod=0;
    double som1_carre=0;
    double som2_carre=0;
    for (unsigned int k=0;k<Ratio_S.size();k++)
    {
         som1_carre= som1_carre+(Ratio_S[k]-moy1)*(Ratio_S[k]-moy1);  //la somme de carré des ratios
         som2_carre= som2_carre+(Ratio_G[k]-moy2)*(Ratio_G[k]-moy2);
         prod=prod+((Ratio_S[k]-moy1)*(Ratio_G[k]-moy2));


    }
    coeff=(prod/sqrt((som1_carre)*(som2_carre)));
    return(coeff);

}


//fournie deux vecteurs de type aSolution: 1/[all_Solution]:toute les combinaisons possibles,leurs coef,l'indice de premier élément ainsi que son dt
                                        // 2/ [solution]:contient la combinaison optimale pour tout les dt possibles(le dt,indice,liste et le coefficient(max))
void S_Appli::coefficient()
{
    std::vector<double> Ratio_Som=calc_Ratios(aSomList);
    dt_max=(aGPSList.size()/aSomList.size());
    for( int dt=1;dt<dt_max+1;dt++)
    {
        for (unsigned int i=0;i<aGPSList.size()-(duree*dt)+1;i++)
        {
            std::vector<Pt3dr> aGPS_sList;
            int j=i;

            for (unsigned int h=0;h<aNameList.size();h++)
            {
                aGPS_sList.push_back(aGPSList[j]);
                j=j+dt*aDecalList[h];

            }
            std::vector<double> Ratio_GPS=calc_Ratios(aGPS_sList);
            double coef_corr=calc_coef(Ratio_Som, Ratio_GPS);
            aSolution sol;
            sol.coefficient=coef_corr;
            sol.indice=i;
            sol.Liste=aGPS_sList;
            sol.t=dt;
            all_Solutions.push_back(sol);
         }
         solution.push_back( cherche_max(all_Solutions,dt));     //cherhe les max de chaque dt

    }

}



// la fonction affiche_max permet l'affichage d'une liste sur le terminale
void S_Appli::affiche_max(std::vector<aSolution> solution , int dt_max)
{
    for(int k=0;k<dt_max;k++)
    {
        std::cout<< "****************** Solution n°"<<k+1<<"******************"<<'\n';
        std::cout<<"dt=" << solution.at(k).t<<'\n';
        std::cout<<"Best Corr Coeff = "<<solution.at(k).coefficient<<'\n';
        std::cout<<"First position = "<< solution.at(k).indice<<'\n';
        //std::cout<<"la solution est"<<'\n';
        //for(unsigned int i=0;i<solution.at(k).Liste.size();i++)
        //{
        //   printf("%.4f  %.4f  %.4f\n", solution.at(k).Liste[i].x, solution.at(k).Liste[i].y, solution.at(k).Liste[i].z);
        //}
        std::cout<< "*************************************************"<<'\n';

    }
}


//la fonction cherche_max fournit la solution optimale pour un dt donnée:
//en trouvant le coefficient le plus proche de 1 et la liste des points GPS qui correspond à ce coefficient ainsi que l'indice du 1er elélément
aSolution S_Appli::cherche_max(std::vector<aSolution> all_Solutions,int dt) 
{

    int a=all_Solutions.size()-1;
    double max=all_Solutions[a].coefficient;
    for(unsigned int i=0;i<all_Solutions.size()-1;i++)
    {
        if (all_Solutions[i].t==dt)
        {
            if (all_Solutions[i].coefficient>max)
                {

                    max=all_Solutions[i].coefficient;

                    a=i;
                }
            }

    }

    return(all_Solutions[a]);
}


//la fonction sauvegarde() permet le sauvegarde de la solution optimale dans un fichier txt 
void S_Appli::sauvegarde()           
{
    int m=0;
    double co_max=solution.at(0).coefficient;
    for(int k=1;k<dt_max;k++)
    {
        if(solution.at(k).coefficient>co_max)
        {
            co_max=solution.at(k).coefficient;
            m=k;
         }
    }
    best = solution.at(m);
    FILE * outfile;
    outfile = fopen(aOut.c_str(), "w");
    //fprintf(outfile,"dt=%d   le 1er indice=%.d   le coefficient=%.6f\n",best.t,best.indice,best.coefficient);
    for(std::size_t i = 0; i < best.Liste.size(); i++)
    {
		Pt3dr pt = best.Liste.at(i);
		std::string aName = aNameImgs.at(i);
		//std::cout << "aName =" << aName << std::endl;
        fprintf(outfile, "%s %.4f %.4f %.4f\n", aName.c_str(), pt.x, pt.y, pt.z);
	}
	
    fclose(outfile);
}


//la fonction Trie retourne les n(n entrée par l'utilisateurs) meilleurs solutions par ordre décroissant
std::vector<aSolution> S_Appli::Trie( int nbr_sol)
{
    std::vector<aSolution> combinaisons=all_Solutions;
    std::vector<aSolution> trie;


    for (int j=0;j<nbr_sol;j++)
       {

            for( unsigned int i=j;i<combinaisons.size();i++)
            {
                if (combinaisons[i].coefficient>combinaisons[j].coefficient)
                {
                    aSolution val_inter;
                    val_inter=combinaisons[j];
                    combinaisons[j]=combinaisons[i];
                    combinaisons[i]=val_inter;

                }
             }
            trie.push_back(combinaisons[j]);

        }
     return(trie);

}

//la fonction test_dt compare le dt des images en données exif et le dt calculé par le programme
void S_Appli::test_dt(int best_dt)         
{
    std::vector<double> T;
    string dossier="Tmp-MM-Dir";
    std::vector<cXmlDate> aDateList;
    ManC=cInterfChantierNameManipulateur::BasicAlloc(dossier);
    std::list<std::string> Img_xif=ManC->StdGetListOfFile(".*xml");
    bool exist=true;

    for(std::list<std::string>::iterator I=Img_xif.begin();I!=Img_xif.end();I++)  //tester si les données exif contiennet date!!
    {
        cXmlXifInfo aXmlXifInfo=StdGetFromPCP(dossier+"/"+*I,XmlXifInfo);
        if(aXmlXifInfo.Date().IsInit())
        {
            aDateList.push_back(aXmlXifInfo.Date().Val());
        }
        else
        {
            std::cout<<"les données EXif ne contiennent pas le temps"<<'\n';
            exist=false;
            break;
        }
    }

    if(exist)     //si la date existe dans les données exif de tous les images 
    {
        for(unsigned int i=0;i<aDateList.size();i++)
        {
            double temps1=aDateList[i+1].Hour().S()+aDateList[i+1].Hour().M()*60+aDateList[i+1].Hour().H()*3600;
            double temps2=aDateList[i].Hour().S()+aDateList[i].Hour().M()*60+aDateList[i].Hour().H()*3600;
             
            T.push_back((temps1-temps2)/aDecalList[i]);
        }

        int t = T[0];
        bool test_unit_tmp=true;
        for(unsigned int i=1;i<aDateList.size();i++)
        {
             if (T[i]!=t)
             {
                 test_unit_tmp=false;
                 std::cout<<"le delta t entre les images n'est pas coherent!!!!"<<'\n';
                 break;
              }
        }
        if(test_unit_tmp)    //si le decalage entre les images en unité de temps est le même
        {
            if(best_dt==t)
            {
                 std::cout<<"Génial! le dT trouvé correspond au dT qui se trouve dans les données exif"<<'\n';
             }
            else
            {
                std::cout<<"!!le dT trouvé ne correspond pas au dT qui se trouve dans les données exif"<<'\n';
            }
         }
    }
}


//la fonction test_exif() génère le dossier Tmp-MM-Dir s'il n'exixte pas
void S_Appli::test_exif()
{
    string dossier="Tmp-MM-Dir";
    if(ELISE_fp::IsDirectory(dossier))        //test si Tmp-MM-Dir existe
    {
        test_dt(best.t);
    }
    else
    {
        system_call((std::string("mm3d MMXmlXif \"")+mFullName+"\"").c_str());
        test_dt(best.t);
    }
}


int MatchCenters_main( int argc, char ** argv )
{
    S_Appli test_Appli(argc,argv );
    return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
