from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt


def plot_pretrain(it, lr, acc, trn, val, save=None):
    fig = plt.figure(1, figsize=(14, 4))
    ax_lr = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height], 0 <= l,b,w,h <= 1

    # parasite additional axes, share x
    ax_acc = ParasiteAxes(ax_lr, sharex=ax_lr)
    ax_trn = ParasiteAxes(ax_lr, sharex=ax_lr)
    ax_val = ParasiteAxes(ax_lr, sharex=ax_lr)

    # append axes
    ax_lr.parasites.append(ax_acc)
    ax_lr.parasites.append(ax_trn)
    ax_lr.parasites.append(ax_val)

    # invisible right axis of ax_lr
    ax_lr.axis['right'].set_visible(False)
    ax_lr.axis['top'].set_visible(False)
    ax_acc.axis['right'].set_visible(True)
    ax_acc.axis['right'].major_ticklabels.set_visible(True)
    ax_acc.axis['right'].label.set_visible(True)

    # set label for axis
    ax_lr.set_ylabel('lr (1e-5)')
    ax_lr.set_xlabel('iter (K)')
    ax_acc.set_ylabel('acc (%)')
    ax_trn.set_ylabel('trn loss')
    ax_val.set_ylabel('val loss')

    trn_axisline = ax_trn.get_grid_helper().new_fixed_axis
    val_axisline = ax_val.get_grid_helper().new_fixed_axis

    ax_trn.axis['right2'] = trn_axisline(loc='right', axes=ax_trn, offset=(60, 0))
    ax_val.axis['right3'] = val_axisline(loc='right', axes=ax_val, offset=(120, 0))

    fig.add_axes(ax_lr)

    curve_lr, = ax_lr.plot(it, lr, label="lr", color='black')
    curve_acc, = ax_acc.plot(it, acc, label="acc", color='red')
    curve_trn, = ax_trn.plot(it, trn, label="trn loss", color='lightblue')
    curve_val, = ax_val.plot(it, val, label="val loss", color='blue')

    ax_acc.set_ylim(17, 21)
    ax_trn.set_ylim(2.4, 2.8)
    ax_val.set_ylim(2.4, 2.8)
    ax_lr.set_ylim(1, 20)

    ax_lr.legend()

    # ax_lr.axis['left'].label.set_color(ax_lr.get_color())
    ax_acc.axis['right'].label.set_color('red')
    ax_trn.axis['right2'].label.set_color('lightblue')
    ax_val.axis['right3'].label.set_color('blue')

    ax_acc.axis['right'].major_ticks.set_color('red')
    ax_trn.axis['right2'].major_ticks.set_color('lightblue')
    ax_val.axis['right3'].major_ticks.set_color('blue')

    ax_acc.axis['right'].major_ticklabels.set_color('red')
    ax_trn.axis['right2'].major_ticklabels.set_color('lightblue')
    ax_val.axis['right3'].major_ticklabels.set_color('blue')

    ax_acc.axis['right'].line.set_color('red')
    ax_trn.axis['right2'].line.set_color('lightblue')
    ax_val.axis['right3'].line.set_color('blue')
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_deepre(ep, loss, acc, save=None):
    plt.subplot(211)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(ep, loss)
    plt.subplot(212)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(ep, acc)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_ecpred(ep, loss, acc, pr, f1, save=None):
    plt.subplot(221)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(ep, loss)
    plt.subplot(222)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(ep, acc)
    plt.subplot(223)
    plt.xlabel('epoch')
    plt.ylabel('P-R')
    plt.plot(ep, pr)
    plt.subplot(224)
    plt.xlabel('epoch')
    plt.ylabel('macro-F1')
    plt.plot(ep, f1)
    if save is not None:
        plt.savefig(save)
    plt.show()
