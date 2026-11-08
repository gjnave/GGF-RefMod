"""Recycle a single file using Windows Explorer's recoverable delete operation."""
def recycle_video(path):
    try:
        import pythoncom, pywintypes
        from win32com.shell import shell, shellcon
        from win32com.server.exception import COMException
        from win32com.server.policy import DesignatedWrapPolicy
    except ImportError as exc:
        raise OSError('Recycle Bin support is unavailable. Run INSTALL.bat to install it.') from exc

    class RecycleOnly(DesignatedWrapPolicy):
        _com_interfaces_=[shell.IID_IFileOperationProgressSink]
        _public_methods_=['StartOperations','FinishOperations','PreRenameItem','PostRenameItem',
                          'PreMoveItem','PostMoveItem','PreCopyItem','PostCopyItem','PreDeleteItem',
                          'PostDeleteItem','PreNewItem','PostNewItem','UpdateProgress','ResetTimer',
                          'PauseTimer','ResumeTimer']
        recycled=False

        def __init__(self):
            self._wrap_(self)

        def PreDeleteItem(self,flags,item):
            if not flags & shellcon.TSF_DELETE_RECYCLE_IF_POSSIBLE:
                raise COMException(desc='Windows cannot recycle this file.',scode=-2147467260)  # E_ABORT

        def PostDeleteItem(self,flags,item,result,new_item):
            self.recycled=result>=0 and new_item is not None

        def __getattr__(self,name):
            if name in self._public_methods_:return lambda *args:None
            raise AttributeError(name)

    pythoncom.CoInitialize()
    try:
        op=pythoncom.CoCreateInstance(shell.CLSID_FileOperation,None,pythoncom.CLSCTX_ALL,shell.IID_IFileOperation)
        op.SetOperationFlags(0x00080000 | shellcon.FOFX_EARLYFAILURE | shellcon.FOF_NOERRORUI |
                             shellcon.FOF_NOCONFIRMATION | shellcon.FOF_SILENT)  # FOFX_RECYCLEONDELETE
        sink=RecycleOnly()
        item=shell.SHCreateItemFromParsingName(str(path),None,shell.IID_IShellItem)
        op.DeleteItem(item,pythoncom.WrapObject(sink,shell.IID_IFileOperationProgressSink))
        op.PerformOperations()
        if op.GetAnyOperationsAborted() or not sink.recycled:
            raise OSError('Windows did not confirm a move to the Recycle Bin.')
    except pywintypes.com_error as exc:
        raise OSError(f'Windows could not recycle this file: {exc.strerror}') from exc
    finally:
        pythoncom.CoUninitialize()
