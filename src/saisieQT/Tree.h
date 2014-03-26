#ifndef TREE_H
#define TREE_H

#include <QAbstractTableModel>
#include <QSortFilterProxyModel>
#include <QColor>

#include "QT_interface_Elise.h"
#include "../uti_phgrm/SaisiePts/SaisiePts.h"

class ModelPointGlobal : public QAbstractTableModel
{
    Q_OBJECT
public:

    ModelPointGlobal(QObject *parent, cAppli_SaisiePts* appli);

    int             rowCount(const QModelIndex &parent = QModelIndex()) const ;

    int             columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant        data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    QVariant        headerData(int section, Qt::Orientation orientation, int role) const;

    bool            setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    Qt::ItemFlags   flags(const QModelIndex &index) const;

    bool            insertRows(int row, int count, const QModelIndex & parent = QModelIndex());

    bool            caseIsSaisie(int idRow);

signals:

    void            pGChanged();

protected:

    int             AllCount() const;

    int             PG_Count() const;

    int             CaseNamePointCount() const;

private:

    cAppli_SaisiePts* mAppli;

};


class ImagesSortFilterProxyModel : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    ImagesSortFilterProxyModel(QObject *parent = 0): QSortFilterProxyModel(parent){}

//    QDate filterMinimumDate() const { return minDate; }
//    void setFilterMinimumDate(const QDate &date);

//    QDate filterMaximumDate() const { return maxDate; }
//    void setFilterMaximumDate(const QDate &date);

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;
   // bool lessThan(const QModelIndex &left, const QModelIndex &right) const;

private:
//    bool dateInRange(const QDate &date) const;

//    QDate minDate;
//    QDate maxDate;
};

class cQT_Interface ;

class ModelCImage : public QAbstractTableModel
{
    Q_OBJECT
public:

    ModelCImage(QObject *parent, cAppli_SaisiePts* appli);

    int             rowCount(const QModelIndex &parent = QModelIndex()) const ;

    int             columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant        data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    QVariant        headerData(int section, Qt::Orientation orientation, int role) const;

    bool            setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    Qt::ItemFlags   flags(const QModelIndex &index) const;

    bool            insertRows(int row, int count, const QModelIndex & parent = QModelIndex());

    int             getIdGlobSelect() const;

    void            setIdGlobSelect(int value);

private:

    cAppli_SaisiePts* mAppli;

    cQT_Interface*  _interface;

    int             idGlobSelect;

};

#endif // TREE_H
