#include "StdAfx.h"
#include "general/CMake_defines.h"
#if ELISE_QT
    #ifdef Int
        #undef Int
    #endif
    #include "QCoreApplication"
    #include "QStringList"
    #include "QDir"
    
    #include "XML_GEN/all.h"
 
    using namespace std;

    string MMQtLibraryPath()
    {
        #if defined(__APPLE__) || defined(__MACH__)
            return MMDir() + "Frameworks";
        #elif ELISE_windows
            return MMBin();
        #endif
        return string();
    }

    // there is alway one path in the list to avoid multiple library loading
    void setQtLibraryPath(const string &i_path)
    {
	    QString path( i_path.c_str() );
	    if ( !QDir(path).exists() ) cerr << "WARNING: setQtLibraryPath(" << i_path << "): path does not exist" << endl;

        QCoreApplication::setLibraryPaths( QStringList(path) );
        // Sometimes the setLibraryPaths change the decimal-point character according the local OS config
        // to be sure that atof("2.5") is always 2.5 it's necessary to force setLocale
        setlocale(LC_NUMERIC, "C");
    }

    // if default path does not exist, replace it by deployment path
    // used by mm3d and SaisieQT
    void initQtLibraryPath()
    {
        // set to install plugins directory if it exists
        const string installPlugins = QT_INSTALL_PLUGINS;
        if ( !installPlugins.empty() && QDir( QString(installPlugins.c_str())).exists() )
        {
            setQtLibraryPath(installPlugins);
            return;
        }

        // set to deployment path if it exists
        const string deploymentPath = MMQtLibraryPath();

        if ( !deploymentPath.empty() && QDir( QString(deploymentPath.c_str())).exists() )
        {
            setQtLibraryPath(deploymentPath);
            return;
        }

        // keep the first existing path to avoid multiple library loading
        QStringList paths = QCoreApplication::libraryPaths();
        for ( int i=0; i<paths.size(); i++ )
        {
            if ( QDir( paths.at(i) ).exists() )
            {
                setQtLibraryPath( paths.at(i).toStdString() );
                return;
            }
        }

        cerr << "WARNING: initQtLibraryPath: no valid path found" << endl;
    }
#endif // ELISE_QT
