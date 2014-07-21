#include "StdAfx.h"

#if (ELISE_QT_VERSION >= 4)
#ifdef Int
#undef Int
#endif
#endif 

class Dimap{
public:
    Dimap(std::string const &nomFile)
    {
        lireDimapFile(nomFile);
    }
    
    void lireDimapFile(std::string const &nomFile);
    
    Pt2dr direct(Pt2dr Pimg, double altitude)const;
    
    Pt2dr indirect(Pt2dr Pgeo, double altitude)const;
    
    Pt2dr ptGeo2Carto(Pt2dr Pgeo, std::string targetSyst)const;
    
    void createDirectGrid(double ulcSamp, double ulcLine,
                          double stepPixel,
                          int nbSamp, int  nbLine,
                          std::vector<double> const &vAltitude,
                          std::vector<Pt2dr> &vPtCarto, std::string targetSyst)const;
    
    void createIndirectGrid(double ulcX, double ulcY,
                            int nbrSamp, int nbrLine,
                            double stepCarto,
                            std::vector<double> const &vAltitude,
                            std::vector<Pt2dr> &vPtImg, std::string targetSyst)const;
    
    
    
    void createGrid(std::string const &nomGrid, std::string const &nomImage,
                    double stepPixel, double stepCarto,
                    double rowCrop, double sampCrop,
                    std::vector<double> vAltitude, std::string targetSyst)const;
    void info()
    {
        std::cout << "long_scale   : "<<long_scale<<  " | long_off   : "<<long_off<<std::endl;
        std::cout << "lat_scale    : "<<lat_scale<<   " | lat_off    : "<<lat_off<<std::endl;
        std::cout << "height_scale : "<<height_scale<<" | height_off : "<<height_off<<std::endl;
        std::cout << "samp_scale   : "<<samp_scale<<  " | samp_off   : "<<samp_off<<std::endl;
        std::cout << "line_scale   : "<<line_scale<<  " | line_off   : "<<line_off<<std::endl;
        std::cout << "first_row    : "<<first_row<<   " | last_row   : "<<last_row<<std::endl;
        std::cout << "first_col    : "<<first_col<<   " | last_col   : "<<last_col<<std::endl;
        std::cout << "first_lon    : "<<first_lon<<   " | last_lon   : "<<last_lon<<std::endl;
        std::cout << "first_lat    : "<<first_lat<<   " | last_lat   : "<<last_lat<<std::endl;
    }
    
    void clearing(std::string const &nomImage)
    {
        std::string aString[] ={nomImage+".GRI",nomImage+".GRC",
            "processing/conv_ptGeo.txt","processing/conv_ptLamb93.txt",
            "processing/direct_ptGeo.txt", "processing/direct_ptLamb93.txt",
            "processing/indirect_ptGeo.txt","processing/indirect_ptLamb93.txt"};
        
        for (int j=0; j<sizeof(aString)/sizeof(aString[0]); j++)
            {
            remove (aString[j].c_str());
            }
        remove ("processing");
    }       

//private:
    std::vector<double> direct_samp_num_coef;
    std::vector<double> direct_samp_den_coef;
    std::vector<double> direct_line_num_coef;
    std::vector<double> direct_line_den_coef;
    
    std::vector<double> indirect_samp_num_coef;
    std::vector<double> indirect_samp_den_coef;
    std::vector<double> indirect_line_num_coef;
    std::vector<double> indirect_line_den_coef;
    
    double indirErrBiasRow;
    double indirErrBiasCol;
    double dirErrBiasX;
    double dirErrBiasY;

    double first_row;
    double first_col;
    double last_row;
    double last_col;
    double first_lon;
    double first_lat;
    double last_lon;
    double last_lat;
    
    double long_scale;
    double long_off;
    double lat_scale;
    double lat_off;

    double samp_scale;
    double samp_off;
    double line_scale;
    double line_off;

    double height_scale;
    double height_off;
    
    
};

Pt2dr Dimap::direct(Pt2dr Pimg, double altitude)const
{
    //Calcul des coordonnées image normalisées
    double Y=(Pimg.y-line_off)/line_scale;
    double X=(Pimg.x-samp_off)/samp_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,X,Y,Z,Y*X,X*Z,Y*Z,X*X,Y*Y,Z*Z,X*Y*Z,X*X*X,Y*Y*X,X*Z*Z,X*X*Y,Y*Y*Y,Y*Z*Z,X*X*Z,Y*Y*Z,Z*Z*Z};
    
    double long_den = 0.;
    double long_num = 0.;
    double lat_den = 0.;
    double lat_num = 0.;
    
    for (int i=0; i<20; i++)
    {
        lat_num  += vecteurD[i]*direct_line_num_coef[i];
        lat_den  += vecteurD[i]*direct_line_den_coef[i];
        long_num += vecteurD[i]*direct_samp_num_coef[i];
        long_den += vecteurD[i]*direct_samp_den_coef[i];
    }
    
    //Calcul final
    Pt2dr Pgeo;
    if ((lat_den != 0) &&
        (long_den !=0))
    {
        Pgeo.x = (lat_num / lat_den) * lat_scale + lat_off;
        Pgeo.y = (long_num / long_den) * long_scale + long_off;
    }
    else
    {
        std::cout << "Erreur de calcul - dénominateur nul"<<std::endl;
    }
    return Pgeo;
}

Pt2dr Dimap::indirect(Pt2dr Pgeo, double altitude)const
{
    //Calcul des coordonnées image normalisées
    double Y=(Pgeo.y-long_off)/long_scale;
    double X=(Pgeo.x-lat_off)/lat_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,Y,X,Z,Y*X,Y*Z,X*Z,Y*Y,X*X,Z*Z,X*Y*Z,Y*Y*Y,Y*X*X,Y*Z*Z,X*Y*Y,X*X*X,X*Z*Z,Y*Y*Z,X*X*Z,Z*Z*Z};
    
    double samp_den = 0.;
    double samp_num = 0.;
    double line_den = 0.;
    double line_num = 0.;
    
    for (int i=0; i<20; i++)
    {
        line_num  += vecteurD[i]*indirect_line_num_coef[i];
        line_den  += vecteurD[i]*indirect_line_den_coef[i];
        samp_num  += vecteurD[i]*indirect_samp_num_coef[i];
        samp_den  += vecteurD[i]*indirect_samp_den_coef[i];
    }
    //Calcul final
    Pt2dr Pimg;
    if ((samp_den != 0) &&
        (line_den !=0))
    {
        Pimg.x = (samp_num / samp_den) * samp_scale + samp_off;
        Pimg.y = (line_num / line_den) * line_scale + line_off;
    }
    else
    {
        std::cout << "Erreur de calcul - dénominateur nul"<<std::endl;
    }
    return Pimg;
}

Pt2dr Dimap::ptGeo2Carto(Pt2dr Pgeo, std::string targetSyst)const
{
    
    std::ofstream fic("processing/conv_ptGeo.txt");
    fic << std::setprecision(15);
    fic << Pgeo.y <<" "<<Pgeo.x<<";"<<std::endl;
    // transfo en Lambert93
    std::string command;
    command = "cs2cs +proj=latlon +datum=WGS84 +ellps=WGS84 +to "+targetSyst+" -s processing/conv_ptGeo.txt > processing/conv_ptLamb93.txt";
    system(command.c_str());
    // chargement des coordonnées du point converti
    Pt2dr PointCarto;
    std::ifstream fic2("processing/conv_ptLamb93.txt");
    while(!fic2.eof()&&fic2.good())
    {
        double X,Y,Z;
        char c;
        fic2 >> Y >> X >> Z >> c;
        if (fic2.good())
            {
                PointCarto.x=X;
                PointCarto.y=Y;
            }
    }
    return PointCarto;
}



void Dimap::createDirectGrid(double ulcSamp, double ulcLine,
                      double stepPixel,
                      int nbSamp, int  nbLine,
                      std::vector<double> const &vAltitude,
                             std::vector<Pt2dr> &vPtCarto, std::string targetSyst)const
{
    vPtCarto.clear();
    // On cree un fichier de point geographique pour les transformer avec proj4
    {
        std::ofstream fic("processing/direct_ptGeo.txt");
        fic << std::setprecision(15);
        for(size_t i=0;i<vAltitude.size();++i)
        {
            double altitude = vAltitude[i];
            for(int l=0;l<nbLine;++l)
            {
                for(int c = 0;c<nbSamp;++c)
                {
                    double cStep = c  * stepPixel;
                    double lStep = l * stepPixel;
                    Pt2dr Pimg(ulcSamp + cStep, ulcLine + lStep);
                    Pt2dr Pgeo = direct(Pimg,altitude);
                    fic << Pgeo.y <<" "<<Pgeo.x<<";"<<std::endl;
                }
            }
        }
    }
    // transfo en Lambert93
    std::string command;
    command = "cs2cs +proj=latlon +datum=WGS84 +ellps=WGS84 +to "+targetSyst+" -s processing/direct_ptGeo.txt >  processing/direct_ptLamb93.txt";
    system(command.c_str());
    // chargement des points
    std::ifstream fic("processing/direct_ptLamb93.txt");
    while(!fic.eof()&&fic.good())
    {
        double X,Y,Z;
        char c;
        fic >> Y >> X >> Z >> c;
        if (fic.good())
            vPtCarto.push_back(Pt2dr(X,Y));
    }
    std::cout << "Nombre de points lus : "<<vPtCarto.size()<<std::endl;
}

void Dimap::createIndirectGrid(double ulcX, double ulcY, int nbrSamp, int nbrLine,
                        double stepCarto,
                        std::vector<double> const &vAltitude,
                               std::vector<Pt2dr> &vPtImg, std::string targetSyst)const
{
    vPtImg.clear();
    // On cree un fichier de point geographique pour les transformer avec proj4
    {
        std::ofstream fic("processing/indirect_ptLamb93.txt");
        fic << std::setprecision(15);
        for(int l=0;l<nbrLine;++l)
        {
            double Y = ulcY - l*stepCarto;
            for(int c = 0;c<nbrSamp;++c)
            {
                double X =ulcX + c*stepCarto;
                fic << X <<" "<<Y<<";"<<std::endl;
            }
        }
    }
    // transfo en Geo
    std::string command;
    command = "cs2cs "+targetSyst+" +to +proj=latlon +datum=WGS84 +ellps=WGS84 -f %.12f -s  processing/indirect_ptLamb93.txt >  processing/indirect_ptGeo.txt";
    system(command.c_str());
    for(size_t i=0;i<vAltitude.size();++i)
    {
        double altitude = vAltitude[i];
        // chargement des points
        std::ifstream fic("processing/indirect_ptGeo.txt");
        while(!fic.eof()&&fic.good())
        {
            double lon ,lat ,Z;
            char c;
            fic >> lat  >> lon >> Z >> c;
            if (fic.good())
            {
                vPtImg.push_back(indirect(Pt2dr(lat,lon),altitude));
            }
        }
    }
    std::cout << "Nombre de points lus : "<<vPtImg.size()<<std::endl;
}



void Dimap::createGrid(std::string const &nomGrid, std::string const &nomImage,
                double stepPixel, double stepCarto,
                double rowCrop, double sampCrop,
                       std::vector<double> vAltitude, std::string targetSyst)const
{
    double firstSamp = first_col;
    double firstLine= first_row;
    double lastSamp = last_col;
    double lastLine = last_row;

    //Creation d'un dossier pour les fichiers intermediaires
    system ("mkdir processing");
    
    //Direct nbr Lignes et colonnes + step last ligne et colonne
    int nbLine, nbSamp;
    nbLine=(lastLine-firstLine)/stepPixel +1;
    nbSamp=(lastSamp-firstSamp)/stepPixel +1 ;
    
    std::vector<Pt2dr> vPtCarto;
    createDirectGrid(firstSamp,firstLine,stepPixel,nbSamp,nbLine,vAltitude,vPtCarto, targetSyst);
    
    double xmin,xmax,ymin,ymax;
    // Pour estimer la zone carto on projette le domaine de validite geo
    // Recherche de la zone la plus etendue
    Pt2dr Pcarto = ptGeo2Carto(Pt2dr(first_lat,first_lon), targetSyst);
    xmin = Pcarto.x;
    ymin = Pcarto.y;
    xmax = xmin;
    ymax = ymin;
    Pcarto = ptGeo2Carto(Pt2dr(first_lat,last_lon), targetSyst);
    if (xmin>Pcarto.x) xmin = Pcarto.x;
    else if (xmax<Pcarto.x) xmax = Pcarto.x;
    if (ymin>Pcarto.y) ymin = Pcarto.y;
    else if (ymax<Pcarto.y) ymax = Pcarto.y;
    Pcarto = ptGeo2Carto(Pt2dr(last_lat,last_lon), targetSyst);
    if (xmin>Pcarto.x) xmin = Pcarto.x;
    else if (xmax<Pcarto.x) xmax = Pcarto.x;
    if (ymin>Pcarto.y) ymin = Pcarto.y;
    else if (ymax<Pcarto.y) ymax = Pcarto.y;
    Pcarto = ptGeo2Carto(Pt2dr(last_lat,first_lon), targetSyst);
    if (xmin>Pcarto.x) xmin = Pcarto.x;
    else if (xmax<Pcarto.x) xmax = Pcarto.x;
    if (ymin>Pcarto.y) ymin = Pcarto.y;
    else if (ymax<Pcarto.y) ymax = Pcarto.y;
    
    //Coins sup gauche et inf droit du domaine de validite RPC inverse
    Pt2dr urc(xmax,ymax);
    Pt2dr llc(xmin,ymin);
    std::cout << "Emprise carto : "<<llc.x<<" "<<urc.y<<" "<<urc.x<<" "<<llc.y<<std::endl;
    
    //Indirect nbr Lignes et colonnes + step last ligne et colonne
    int nbrLine, nbrSamp;
    nbrSamp=(urc.x-llc.x)/stepCarto +1;
    nbrLine=(urc.y-llc.y)/stepCarto +1;

    std::vector<Pt2dr> vPtImg;
    createIndirectGrid( xmin, ymax,nbrSamp,nbrLine,stepCarto,vAltitude,vPtImg, targetSyst);
    
    //Ecriture de la grille
    //Création de la grille et du flux d'ecriture
    std::ofstream writeGrid(nomGrid.c_str());
    writeGrid <<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<std::endl;
    writeGrid <<"<trans_coord_grid version=\"5\" name=\"\">"<<std::endl;
        //création de la date
        time_t t= time(0);
        struct tm * timeInfo =localtime(&t);
        std::string date;
        std::stringstream ssdate;
        ssdate<<timeInfo-> tm_year+1900;
        double adate []= {timeInfo-> tm_mon, timeInfo-> tm_mday,
                timeInfo-> tm_hour,timeInfo-> tm_min, timeInfo-> tm_sec};
        std::vector<double> vdate (adate,adate+5);
        //Mise en forme de la date
        for (int ida=0; ida<5;ida++)
            {
                std::stringstream ssdateTempo;
                std::string dateTempo;
                ssdateTempo<<vdate[ida];
                dateTempo=ssdateTempo.str();
                if (dateTempo.length()==2)
                    ssdate<<dateTempo;
                else ssdate<<0<<dateTempo;
            }
        date=ssdate.str();
        writeGrid <<"\t<date>"<<date<<"</date>"<<std::endl;
    
        writeGrid <<"\t<trans_coord name=\"\">"<<std::endl;
            writeGrid <<"\t\t<trans_sys_coord name=\"\">"<<std::endl;
                writeGrid <<"\t\t\t<sys_coord name=\"sys1\">"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_plani name=\"sys1\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<nomImage<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"p"<<"</unit>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<direct>"<<"0"<<"</direct>"<<std::endl;    
                        writeGrid <<"\t\t\t\t\t<sub_code>"<<"*"<<"</sub_code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<vertical>"<<nomImage<<"</vertical>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_plani>"<<std::endl;   
                    writeGrid <<"\t\t\t\t<sys_coord_alti name=\"sys1\">"<<std::endl;    
                        writeGrid <<"\t\t\t\t\t<code>"<<"LAMBERT93"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_alti>"<<std::endl;    
                writeGrid <<"\t\t\t</sys_coord>"<<std::endl;
    
                writeGrid <<"\t\t\t<sys_coord name=\"sys2\">"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_plani name=\"sys2\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<"LAMBERT93"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<direct>"<<"1"<<"</direct>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<sub_code>"<<"*"<<"</sub_code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<vertical>"<<"LAMBERT93"<<"</vertical>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_plani>"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_alti name=\"sys2\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<"LAMBERT93"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_alti>"<<std::endl;
                writeGrid <<"\t\t\t</sys_coord>"<<std::endl;
            writeGrid <<"\t\t</trans_sys_coord>"<<std::endl;   
            writeGrid <<"\t\t<category>"<<"1"<<"</category>"<<std::endl;
            writeGrid <<"\t\t<type_modele>"<<"2"<<"</type_modele>"<<std::endl;
            writeGrid <<"\t\t<direct_available>"<<"1"<<"</direct_available>"<<std::endl;
            writeGrid <<"\t\t<inverse_available>"<<"1"<<"</inverse_available>"<<std::endl;    
        writeGrid <<"\t</trans_coord>"<<std::endl;
    
        writeGrid <<"\t<multi_grid version=\"1\" name=\"1-2\" >"<<std::endl;
            std::stringstream ssUL;
            std::stringstream ssStepPix;
            std::stringstream ssNumbCol;
            std::stringstream ssNumbLine;
            std::string sUpperLeft;
            std::string sStepPix;
            std::string sNumbCol;
            std::string sNumbLine;
            ssUL<<std::setprecision(15)<<firstSamp-sampCrop<<"  "<<std::setprecision(15)<<firstLine-rowCrop;
            ssStepPix<<stepPixel;
            ssNumbCol<<nbSamp;
            ssNumbLine<<nbLine;
            sUpperLeft=ssUL.str();
            sStepPix=ssStepPix.str();
            sNumbCol=ssNumbCol.str();
            sNumbLine=ssNumbLine.str();
            writeGrid <<"\t\t<upper_left>"<<sUpperLeft<<"</upper_left>"<<std::endl;
            writeGrid <<"\t\t<columns_interval>"<<sStepPix<<"</columns_interval>"<<std::endl;
            writeGrid <<"\t\t<rows_interval>"<<"-"+sStepPix<<"</rows_interval>"<<std::endl;
            writeGrid <<"\t\t<columns_number>"<<sNumbCol<<"</columns_number>"<<std::endl;
            writeGrid <<"\t\t<rows_number>"<<sNumbLine<<"</rows_number>"<<std::endl;
            writeGrid <<"\t\t<components_number>"<<"2"<<"</components_number>"<<std::endl;    
            std::vector<Pt2dr>::const_iterator it = vPtCarto.begin();
    
            for(size_t i=0;i<vAltitude.size();++i)
                {
                        std::stringstream ssAlti;
                        std::string sAlti;
                        ssAlti<<std::setprecision(15)<<vAltitude[i];
                        sAlti=ssAlti.str();
                        writeGrid <<"\t\t\t<layer value=\""<<sAlti<<"\">"<<std::endl;
        
                        for(int l=0;l<nbLine;++l)
                            {
                                    for(int c = 0;c<nbSamp;++c)
                                        {
                                            Pt2dr const &PtCarto = (*it);
                                            ++it;
                                            std::stringstream ssCoord;
                                            std::string  sCoord;
                                            ssCoord<<std::setprecision(15)<<PtCarto.x<<"   "<<std::setprecision(15)<<PtCarto.y;
                                            sCoord=ssCoord.str();
                                            writeGrid <<"\t\t\t"<<sCoord<<std::endl;
                                        }
                            }
                        writeGrid <<"\t\t\t</layer>"<<std::endl;
                }
        writeGrid <<"\t</multi_grid>"<<std::endl;

////pour la grille inverse
        writeGrid <<"\t<multi_grid version=\"1\" name=\"2-1\" >"<<std::endl;
            std::stringstream ssULInv;
            std::stringstream ssStepCarto;
            std::stringstream ssNumbColInv;
            std::stringstream ssNumbLineInv;
            std::string sUpperLeftInv;
            std::string sStepCarto;
            std::string sNumbColInv;
            std::string sNumbLineInv;
            ssULInv<<std::setprecision(15)<<vPtCarto[0].x<<"  "<<std::setprecision(15)<<vPtCarto[0].y;
            ssStepCarto<<std::setprecision(15)<<stepCarto;
            ssNumbColInv<<nbrSamp;
            ssNumbLineInv<<nbrLine;
            sUpperLeftInv=ssULInv.str();
            sStepCarto=ssStepCarto.str();
            sNumbColInv=ssNumbColInv.str();
            sNumbLineInv=ssNumbLineInv.str();
            writeGrid <<"\t\t<upper_left>"<<sUpperLeftInv<<"</upper_left>"<<std::endl;
            writeGrid <<"\t\t<columns_interval>"<<sStepCarto<<"</columns_interval>"<<std::endl;
            writeGrid <<"\t\t<rows_interval>"<<sStepCarto<<"</rows_interval>"<<std::endl;
            writeGrid <<"\t\t<columns_number>"<<sNumbColInv<<"</columns_number>"<<std::endl;
            writeGrid <<"\t\t<rows_number>"<<sNumbLineInv<<"</rows_number>"<<std::endl;
            writeGrid <<"\t\t<components_number>"<<"2"<<"</components_number>"<<std::endl;
            std::vector<Pt2dr>::const_iterator it2 = vPtImg.begin();
    
            for(size_t i=0;i<vAltitude.size();++i)
                {
                    std::stringstream ssAlti;
                    std::string sAlti;
                    ssAlti<<std::setprecision(15)<<vAltitude[i];
                    sAlti=ssAlti.str();
                    writeGrid <<"\t\t\t<layer value=\""<<sAlti<<"\">"<<std::endl;
        
                    for(int l=0;l<nbrLine;++l)
                        {
                            for(int c = 0;c<nbrSamp;++c)
                                {
                                    Pt2dr const &PtImg = (*it2);
                                    ++it2;
                                    std::stringstream ssCoordInv;
                                    std::string  sCoordInv;
                                    ssCoordInv<<std::setprecision(15)<<PtImg.x - sampCrop<<"   "
                                        <<std::setprecision(15)<<PtImg.y - rowCrop;
                                    sCoordInv=ssCoordInv.str();
                                    writeGrid <<"\t\t\t"<<sCoordInv<<std::endl;
                                }
                        }
                    writeGrid <<"\t\t\t</layer>"<<std::endl;
                }
        writeGrid <<"\t</multi_grid>"<<std::endl;
   
    writeGrid <<"</trans_coord_grid>"<<std::endl;
}

//Lecture du fichier DIMAP
void Dimap::lireDimapFile(std::string const &nomFile)
{
    direct_samp_num_coef.clear();
    direct_samp_den_coef.clear();
    direct_line_num_coef.clear();
    direct_line_den_coef.clear();
    
    indirect_samp_num_coef.clear();
    indirect_samp_den_coef.clear();
    indirect_line_num_coef.clear();
    indirect_line_den_coef.clear();

    cElXMLTree tree(nomFile.c_str());
    
 {
    std::list<cElXMLTree*> noeuds=tree.GetAll(std::string("Direct_Model"));
    std::list<cElXMLTree*>::iterator it_grid,fin_grid=noeuds.end();


    std::string coefSampN="SAMP_NUM_COEFF";
    std::string coefSampD="SAMP_DEN_COEFF";
    std::string coefLineN="LINE_NUM_COEFF";
    std::string coefLineD="LINE_DEN_COEFF";
    for (int c=1; c<21;c++)
    {
        double value;
        std::stringstream ss;
        ss<<"_"<<c;
        coefSampN.append(ss.str());
        coefSampD.append(ss.str());
        coefLineN.append(ss.str());
        coefLineD.append(ss.str());
        for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
        {
            double value;
            std::istringstream buffer((*it_grid)->GetUnique(coefSampN.c_str())->GetUniqueVal());
            buffer >> value;
            direct_samp_num_coef.push_back(value);
            std::istringstream buffer2((*it_grid)->GetUnique(coefSampD.c_str())->GetUniqueVal());
            buffer2 >> value;
            direct_samp_den_coef.push_back(value);
            std::istringstream buffer3((*it_grid)->GetUnique(coefLineN.c_str())->GetUniqueVal());
            buffer3 >> value;
             direct_line_num_coef.push_back(value);
            std::istringstream buffer4((*it_grid)->GetUnique(coefLineD.c_str())->GetUniqueVal());
            buffer4 >> value;
            direct_line_den_coef.push_back(value);
        } 
        coefSampN=coefSampN.substr(0,14);
        coefSampD=coefSampD.substr(0,14);
        coefLineN=coefLineN.substr(0,14);
        coefLineD=coefLineD.substr(0,14);
        
    }
     for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
     {
         std::istringstream buffer((*it_grid)->GetUnique("ERR_BIAS_X")->GetUniqueVal());
         buffer >> dirErrBiasX;
         std::istringstream bufferb((*it_grid)->GetUnique("ERR_BIAS_Y")->GetUniqueVal());
         bufferb >> dirErrBiasY;
     }
 }         
        

 {
    std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Inverse_Model"));
    std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();
    
    std::string coefSampN="SAMP_NUM_COEFF";
    std::string coefSampD="SAMP_DEN_COEFF";
    std::string coefLineN="LINE_NUM_COEFF";
    std::string coefLineD="LINE_DEN_COEFF";
    for (int c=1; c<21;c++)
    {
        double value;
        std::stringstream ss;
        ss<<"_"<<c;
        coefSampN.append(ss.str());
        coefSampD.append(ss.str());
        coefLineN.append(ss.str());
        coefLineD.append(ss.str());
        for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
        {
            std::istringstream bufferInd((*it_gridInd)->GetUnique(coefSampN.c_str())->GetUniqueVal());
            bufferInd >> value;
            indirect_samp_num_coef.push_back(value);
            std::istringstream bufferInd2((*it_gridInd)->GetUnique(coefSampD.c_str())->GetUniqueVal());
            bufferInd2 >> value;
            indirect_samp_den_coef.push_back(value);
            std::istringstream bufferInd3((*it_gridInd)->GetUnique(coefLineN.c_str())->GetUniqueVal());
            bufferInd3 >> value;
            indirect_line_num_coef.push_back(value);
            std::istringstream bufferInd4((*it_gridInd)->GetUnique(coefLineD.c_str())->GetUniqueVal());
            bufferInd4 >> value;
            indirect_line_den_coef.push_back(value);
        }
        coefSampN=coefSampN.substr(0,14);
        coefSampD=coefSampD.substr(0,14);
        coefLineN=coefLineN.substr(0,14);
        coefLineD=coefLineD.substr(0,14);
    }
     for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
     {
         std::istringstream buffer((*it_gridInd)->GetUnique("ERR_BIAS_ROW")->GetUniqueVal());
         buffer >> indirErrBiasRow;
        std::istringstream bufferb((*it_gridInd)->GetUnique("ERR_BIAS_COL")->GetUniqueVal());
         bufferb >> indirErrBiasCol;
     }
  }
  
    
  {
        std::list<cElXMLTree*> noeudsRFM=tree.GetAll(std::string("RFM_Validity"));
        std::list<cElXMLTree*>::iterator it_gridRFM,fin_gridRFM=noeudsRFM.end();

    {
        std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Direct_Model_Validity_Domain"));
        std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();

        
        for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
            {
                std::istringstream bufferInd((*it_gridInd)->GetUnique("FIRST_ROW")->GetUniqueVal());
                bufferInd >> first_row;
                std::istringstream bufferInd2((*it_gridInd)->GetUnique("FIRST_COL")->GetUniqueVal());
                bufferInd2 >> first_col;
                std::istringstream bufferInd3((*it_gridInd)->GetUnique("LAST_ROW")->GetUniqueVal());
                bufferInd3 >> last_row;
                std::istringstream bufferInd4((*it_gridInd)->GetUnique("LAST_COL")->GetUniqueVal());
                bufferInd4 >> last_col;
            }
    }
    
    
    {
        std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Inverse_Model_Validity_Domain"));
        std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();

        for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
        {
            std::istringstream bufferInd((*it_gridInd)->GetUnique("FIRST_LON")->GetUniqueVal());
            bufferInd >> first_lon;
            std::istringstream bufferInd2((*it_gridInd)->GetUnique("FIRST_LAT")->GetUniqueVal());
            bufferInd2 >> first_lat;
            std::istringstream bufferInd3((*it_gridInd)->GetUnique("LAST_LON")->GetUniqueVal());
            bufferInd3 >> last_lon;
            std::istringstream bufferInd4((*it_gridInd)->GetUnique("LAST_LAT")->GetUniqueVal());
            bufferInd4 >> last_lat;
        }
    }
      for(it_gridRFM=noeudsRFM.begin();it_gridRFM!=fin_gridRFM;++it_gridRFM)
      {
          std::istringstream buffer((*it_gridRFM)->GetUnique("LONG_SCALE")->GetUniqueVal());
          buffer>> long_scale;
          std::istringstream buffer2((*it_gridRFM)->GetUnique("LONG_OFF")->GetUniqueVal());
          buffer2 >> long_off;
          std::istringstream buffer3((*it_gridRFM)->GetUnique("LAT_SCALE")->GetUniqueVal());
          buffer3 >> lat_scale;
          std::istringstream buffer4((*it_gridRFM)->GetUnique("LAT_OFF")->GetUniqueVal());
          buffer4 >> lat_off;
          std::istringstream buffer5((*it_gridRFM)->GetUnique("HEIGHT_SCALE")->GetUniqueVal());
          buffer5 >> height_scale;
          std::istringstream buffer6((*it_gridRFM)->GetUnique("HEIGHT_OFF")->GetUniqueVal());
          buffer6 >> height_off;
          std::istringstream buffer7((*it_gridRFM)->GetUnique("SAMP_SCALE")->GetUniqueVal());
          buffer7 >> samp_scale;
          std::istringstream buffer8((*it_gridRFM)->GetUnique("SAMP_OFF")->GetUniqueVal());
          buffer8 >> samp_off;
          std::istringstream buffer9((*it_gridRFM)->GetUnique("LINE_SCALE")->GetUniqueVal());
          buffer9 >> line_scale;
          std::istringstream buffer10((*it_gridRFM)->GetUnique("LINE_OFF")->GetUniqueVal());
          buffer10 >> line_off;
      }
 }

}



int Dimap2Grid_main(int argc, char **argv) {
    std::string aNameFileDimap;// fichier Dimap
    std::string aNameFileGrid;// fichier GRID
    std::string aNameImage;//nom de l'image traitee
    std::string targetSyst="+init=IGNF:LAMB93";//systeme de projection cible - format proj4
    double altiMin;
    double altiMax;
    int nbLayers;

    double stepPixel = 100;
    double stepCarto = 50;
    
    int rowCrop = 0;
    int sampCrop = 0;
    
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileDimap,"Dimap file")
                   << EAMC(aNameFileGrid,"Grid file")
                   << EAMC(aNameImage,"Image name")
                   << EAMC(altiMin,"altitude min")
                   << EAMC(altiMax,"altitude max")
                   << EAMC(nbLayers,"number of layers")
     ,
     LArgMain()
     //caractéristique du système géodésique saisies sans espace (+proj=utm+zone=10+datum=NAD83...)
     << EAM(targetSyst,"targetSyst", true,"target system Proj4 +init=IGNF:LAMB93+datum")
     << EAM(stepPixel,"stepPixel",true,"Step in pixel")
     << EAM(stepCarto,"stepCarto",true,"Step in m (carto)")
     << EAM(sampCrop,"sampCrop",true,"upper left samp - crop")
     << EAM(rowCrop,"rowCrop",true,"upper left row - crop")
     
     );
    
    Dimap dimap(aNameFileDimap);
    std::cout << "Dimap info:"<<std::endl;
    std::cout << "=============================="<<std::endl;
    dimap.info();
    std::cout << "=============================="<<std::endl;
    
    dimap.clearing(aNameImage);
    
    std::vector<double> vAltitude;
    for(int i=0;i<nbLayers;++i)
        vAltitude.push_back(altiMin+i*(altiMax-altiMin)/(nbLayers-1));
    
    //Parser du targetSyst
    //std::string systParsed=targetSyst;
    std::size_t found = targetSyst.find_first_of("+");
    std::string str="+";
    std::vector<int> position;
    while (found!=std::string::npos)
    {
        targetSyst[found]=' ';
        position.push_back(found);
        found=targetSyst.find_first_of("+",found+1);
    }
    for (int i=position.size()-1; i>-1;i--)
        targetSyst.insert(position[i]+1,str);
    
    dimap.createGrid(aNameFileGrid,aNameImage,
                     stepPixel,stepCarto,
                     rowCrop, sampCrop,
                     vAltitude,targetSyst);
    
    return EXIT_SUCCESS;
}

