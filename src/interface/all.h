#ifndef ALL_H
#define ALL_H

#ifdef _WIN32
	#define Q_WS_WIN
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ostream>
#include <typeinfo>
#include <utility>
#if defined Q_WS_WIN
	#define NOMINMAX
	#include <windows.h>
#elif defined Q_WS_MAC
    #include <sys/types.h>
#endif

#ifdef ELISE_Darwin
	#include "OpenGL/gl.h"
	#include "OpenGL/glu.h"	
#else
	#include "GL/gl.h"
	#include "GL/glu.h"
#endif

#include <QtOpenGL>
#include <QtPlugin>
#include <QLabel>
#include <QGLWidget>
#include <QString>
#include <QApplication>
#include <QSettings>
#include <QFileDialog>
#include <QLineEdit>
#include <QDialog>
#include <QDialogButtonBox>
#include <QBoxLayout>
#include <QLocale>
#include <QTranslator>
#include <QtPlugin>
#include <QComboBox>
#include <QProgressDialog>
#include <QToolButton>
#include <QButtonGroup>
#include <QTreeWidget>
#include <QSpinBox>
#include <QInputDialog>
#include <QByteArray>
#include <QDir>
#include <QLibraryInfo>
#include <QProcess>
#include <QMessageBox>
#include <QPainter>
#include <QPolygon>
#include <QWidget>
#include <QMainWindow>
#include <QBrush>
#include <QPen>
#include <QPixmap>
#include <QPoint>
#include <QList>
#include <QTime>
#include <QPushButton>
#include <QRadioButton>
#include <QListWidget>
#include <QTextEdit>
#include <QGroupBox>
#include <QTableWidget>
#include <QCheckBox>
#include <QtAlgorithms>
#include <QHBoxLayout>
#include <QFile>
#include <QTextStream>
#include <QVariant>
#include <QStringList>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>
#include <QThread>
#include <QGridLayout>
#include <QFileInfo>
#include <QTabWidget>
#include <QPaintEvent>
#include <QAction>
#include <QCursor>
#include <QToolBox>
#include <QChar>
#include <QPalette>
#include <QScrollArea>
#include <QDebug>
#include <QDesktopWidget>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QFormLayout>
#include <QSignalMapper>
#include <QHeaderView>
#include <QMenu>
#include <QMenuBar>
#include <QTextCodec>
#include <QtGlobal>

#include "StdAfx.h"

class ParamMain;
class CalibCam;
class ParamImage;
class UserOrientation;
class CarteDeProfondeur;
class GeorefMNT;
class VueChantier;
class Interface;
class RotationButton;
class ParamMasqueXml;
class FileDialog;
class Assistant;
class InterfOptions;
class InterfVerifMicmac;
class MasqueWidget;
class DirectionWidget;
class EchelleWidget;
class ParamPastis;
class PaintInterf;
class PaintInterfSegment;
class PaintInterfAppui;
class VueHomologues;
class InterfModele3D;
class InterfOrtho;
class AppliThread;
class Progression;

#include "assistant.h"
#include "drawMask.h"
#include "readwrite.h"
#include "appliThread.h"
#include "interfPastis.h"
#include "interfApero.h"
#include "interfMicmac.h"
#include "interfConversion.h"
#include "vueChantier.h"
#include "progressBar.h"
#include "interface.h"


#endif
