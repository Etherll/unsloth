# SPDX-License-Identifier: AGPL-3.0-only
# Read only this probe's own child. Never emit or retain its full environment.
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class DebugEnvironment {
 [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr OpenProcess(uint a,bool inherit,int pid);
 [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool ReadProcessMemory(IntPtr h,IntPtr p,byte[] b,int n,out IntPtr read);
 [DllImport("ntdll.dll")] static extern int NtQueryInformationProcess(IntPtr h,int c,IntPtr[] info,int n,out int length);
 static byte[] Read(IntPtr h,long address,int count) {
  var bytes=new byte[count]; IntPtr read;
  if(!ReadProcessMemory(h,new IntPtr(address),bytes,count,out read)||read.ToInt64()!=count) throw new Exception("Environment read failed: "+Marshal.GetLastWin32Error());
  return bytes;
 }
 public static string ReadFlag(int pid) {
  if(IntPtr.Size!=8) throw new Exception("x64 probe required");
  IntPtr h=OpenProcess(0x410,false,pid);
  if(h==IntPtr.Zero) throw new Exception("Own child open failed");
  try {
   var info=new IntPtr[6];int length;
   if(NtQueryInformationProcess(h,0,info,48,out length)!=0 || info[4].ToInt64()!=pid) throw new Exception("Child PEB query failed");
   long parameters=BitConverter.ToInt64(Read(h,info[1].ToInt64()+0x20,8),0);
   long environment=BitConverter.ToInt64(Read(h,parameters+0x80,8),0);
   var entry=new StringBuilder();
   for(int offset=0;offset<131072;offset+=2) {
    char ch=(char)BitConverter.ToUInt16(Read(h,environment+offset,2),0);
    if(ch!='\0') {entry.Append(ch);continue;}
    if(entry.Length==0) return "<absent>";
    const string key="WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=";
    if(entry.ToString().StartsWith(key,StringComparison.OrdinalIgnoreCase)) return entry.ToString().Substring(key.Length);
    entry.Clear();
   }
   return "<bounded scan exhausted>";
  } finally {CloseHandle(h);}
 }
}
'@
