#ifndef ELISE_QT_H
#define ELISE_QT_H

#include "general/CMake_defines.h"

#include <string>

using namespace std;

#if (ELISE_windows & ELISE_MinGW)
    #include "QTCore/qt_windows.h"
    #undef MemoryBarrier
#endif

#ifdef _WIN32
#define NOMINMAX
#include "windows.h"
#endif

#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
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
#include <QtConcurrent/QtConcurrentRun>
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

#include <QOpenGLFunctions>
#include <QOpenGLContext>

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
#include <QStyledItemDelegate>
#include <QSortFilterProxyModel>
#include <QUrl>
#include <QVector>
#include <QXmlStreamReader>
#include <QWidget>
#include <QtGlobal>

#define M_2PI	6.283185307179586232

#endif // ELISE_QT_H
