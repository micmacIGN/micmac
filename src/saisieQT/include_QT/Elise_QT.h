#ifndef ELISE_QT_H
#define ELISE_QT_H

#include "general/CMake_defines.h"


#if (ELISE_windows & ELISE_MinGW)
    #include "QTCore/qt_windows.h"
    #undef MemoryBarrier
#endif

#ifdef _WIN32
#define NOMINMAX
#include "windows.h"
#endif

#if ELISE_QT_VERSION == 4
#include <gl_core_2_1.h>
#endif

#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
	#include <GL/gl.h>
#ifdef _WIN32
	#if ELISE_QT_VERSION == 5
		
	#endif
#endif
#endif

#ifdef Int
    #undef Int
#endif

#include <QAction>
#include <QApplication>
#include <QAbstractTableModel>
#include <QColor>
#include <QCheckBox>
#include <QComboBox>
#if ELISE_QT_VERSION == 5
#include <QtConcurrent/QtConcurrentRun>
#elif ELISE_QT_VERSION == 4
#include <QtConcurrentRun>
#endif
#include <QDebug>
#include <QDir>
#include <QDomDocument>
#include "qdesktopwidget.h"
#include <QFutureWatcher>
#include <QDesktopWidget>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileDialog>
#include <QFlags>
#include <QGLWidget>
#include <QGLShaderProgram>
#include <QGLContext>
#include <QGLBuffer>

#if ELISE_QT_VERSION == 5
#include <qopenglfunctions>
#include <QOpenGLContext>
#endif


#include <QGridLayout>
#include <QtGui>
#include <QIcon>

#include <QImage>
#include <QImageReader>
#include <QInputDialog>
#include "qiodevice.h"
#include <QLabel>
#include <QLineEdit>
#include <QtGui/QMouseEvent>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPoint>
#include <QProgressDialog>
#include <QProcess>
#include <QPushButton>
#include <QTextStream>
#include <QTime>
#include <QTimer>
#include <QTableView>
#include <QTreeView>
#include <QToolBox>
#include <QSettings>
#include <QShortcut>
#include <QSignalMapper>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStyle>
#include <QSortFilterProxyModel>
#include <QUrl>
#include <QVector>
#include <QXmlStreamReader>
#include <QWidget>


#endif // ELISE_QT_H
