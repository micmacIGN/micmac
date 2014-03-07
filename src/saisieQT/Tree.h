#ifndef TREE_H
#define TREE_H

#include <QStandardItemModel>
#include <QFontMetrics>

#include "StdAfx.h"

using namespace NS_SaisiePts;

class TreeItem : public QStandardItem
{
public:
    explicit TreeItem(const QList<QVariant> &data, TreeItem *parent = 0);
    ~TreeItem();

    void appendChild(TreeItem *child);

    TreeItem *child(int row);
    int childCount() const;
    int columnCount() const;
    QVariant data(int column) const;
    void setData(const QVariant &value, int role = Qt::UserRole + 1);

    void setData(const QList<QVariant> &data) { itemData = data; }

    int row() const;
    TreeItem *parent();

private:
    QList<TreeItem*> childItems;
    QList<QVariant> itemData;
    TreeItem *parentItem;
};

class TreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    explicit TreeModel(QObject *parent = 0);
    ~TreeModel();

    QVariant        data(const QModelIndex &index, int role) const;
    bool            setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);
    Qt::ItemFlags   flags(const QModelIndex &index) const;
    QVariant        headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
    QModelIndex     index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex     parent(const QModelIndex &index) const;

    int             rowCount    (const QModelIndex &parent = QModelIndex()) const;
    int             columnCount (const QModelIndex &parent = QModelIndex()) const;

    void            setAppli(cAppli_SaisiePts* appli);

    void            setupModelData();

    void            updateData();

    QList<QVariant> buildRow(cSP_PointGlob *aPG);

    QList<QVariant> buildChildRow(std::pair<std::string, cSP_PointeImage *> data);

    int             getColumnSize(int column, QFontMetrics fm);

private:
    TreeItem *rootItem;

    cAppli_SaisiePts* _appli;
};

#endif // TREE_H
