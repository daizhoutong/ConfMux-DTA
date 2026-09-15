#!/usr/bin/env python3
"""Download only a selected public ConfMux-DTA resource. Python >=3.9, stdlib only.

No login, draft API, training, model loading or automatic package installation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DATA_DIR=Path(__file__).resolve().parent
SOURCES=json.loads((DATA_DIR/'sources.json').read_text(encoding='utf-8'))

def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def request(url,timeout=45):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ('zenodo.org','www.zenodo.org') or '/draft/' in parsed.path:
        raise ValueError('Only public HTTPS Zenodo URLs are supported.')
    r=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ConfMux-DTA-data-helper/20260915'}),timeout=timeout)
    final=urllib.parse.urlparse(r.url)
    if final.scheme!='https' or final.hostname not in ('zenodo.org','www.zenodo.org') or '/draft/' in final.path:
        r.close();raise ValueError('Unexpected non-public download redirect.')
    return r

def public_file(source):
    try:
        with request('https://zenodo.org/api/records/'+source['record_id']) as r:
            record=json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401,403,404):
            raise RuntimeError('Public record unavailable. Open https://doi.org/'+source['doi']+' and check service availability and publication status. A reserved DOI alone is not a public download; publish the deposit first if it is still a draft. No private draft URL will be used.') from e
        raise
    files=record.get('files',[])
    if isinstance(files,dict):files=list(files.get('entries',{}).values())
    matches=[f for f in files if f.get('key',f.get('name'))==source['filename']]
    if len(matches)!=1:raise RuntimeError('Expected filename not found in public record: '+source['filename'])
    entry=matches[0]
    checksum=entry.get('checksum','')
    if not checksum or ':' not in checksum:raise RuntimeError('Public file has no usable checksum; refusing unchecked download.')
    algorithm,value=checksum.split(':',1)
    if algorithm not in ('md5','sha256','sha512'):raise RuntimeError('Unsupported checksum algorithm: '+algorithm)
    return entry,algorithm,value

def verify_file(path,source,entry=None,algorithm=None,value=None):
    if source.get('bytes') is not None and path.stat().st_size!=source['bytes']:
        raise RuntimeError('File size differs from the fixed revision archive.')
    if entry and entry.get('size') is not None and path.stat().st_size!=entry['size']:
        raise RuntimeError('Downloaded size differs from Zenodo metadata.')
    if source.get('sha256') and digest(path)!=source['sha256']:
        raise RuntimeError('SHA256 differs from the fixed revision archive. Do not silently update the expected hash.')
    if algorithm and digest(path,algorithm)!=value:
        raise RuntimeError('Checksum differs from Zenodo metadata.')
    if not source.get('sha256') and not algorithm:
        raise RuntimeError('No trusted checksum is available for this local file.')

def download(source,outdir):
    entry,algorithm,value=public_file(source)
    outdir.mkdir(parents=True,exist_ok=True)
    target=outdir/source['filename']
    if target.exists():
        verify_file(target,source,entry,algorithm,value)
        print('Verified existing file:',target);return target
    url='https://zenodo.org/records/'+source['record_id']+'/files/'+urllib.parse.quote(source['filename'])+'?download=1'
    partial=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix=source['filename']+'.',suffix='.part',dir=outdir,delete=False) as f:
            partial=Path(f.name)
            with request(url) as r:
                count=0;last=0
                while True:
                    block=r.read(4*1024*1024)
                    if not block:break
                    f.write(block);count+=len(block)
                    if count-last>=64*1024*1024:
                        print('Downloaded %.0f MiB'%(count/1024**2),flush=True);last=count
        verify_file(partial,source,entry,algorithm,value)
        if target.exists():raise RuntimeError('Destination appeared during download; refusing to overwrite.')
        # Exclusive creation works on Windows and Unix; no concurrent file is replaced.
        try:
            with target.open('xb') as out,partial.open('rb') as inp:shutil.copyfileobj(inp,out,4*1024*1024)
        except BaseException:
            # Keep a failed final copy for explicit inspection instead of overwriting it later.
            raise
        print('Downloaded and verified:',target)
        return target
    finally:
        if partial is not None and partial.exists():partial.unlink()

def safe_extract(archive,destination):
    destination=destination.resolve()
    if destination.exists():raise RuntimeError('Extraction destination already exists; choose a new directory or use its existing files: '+str(destination))
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if len(members)>20000 or sum(x.file_size for x in members)>30*1024**3:
            raise RuntimeError('Archive exceeds extraction limits.')
        seen=set()
        for info in members:
            name=info.filename
            p=PurePosixPath(name)
            if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
                raise RuntimeError('Unsafe archive path: '+name)
            if stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Archive symlink rejected: '+name)
            key=name.rstrip('/').casefold()
            if key in seen:raise RuntimeError('Duplicate archive path: '+name)
            seen.add(key)
            if not (destination/Path(*p.parts)).resolve().is_relative_to(destination):
                raise RuntimeError('Archive path escapes extraction directory.')
        fileparts=[PurePosixPath(i.filename).parts for i in members if not i.is_dir()]
        roots={parts[0] for parts in fileparts if parts}
        strip_root=len(roots)==1 and bool(fileparts) and all(len(parts)>1 for parts in fileparts)
        destination.mkdir(parents=True)
        for info in members:
            parts=PurePosixPath(info.filename).parts
            if strip_root:parts=parts[1:]
            if not parts:continue
            target=destination/Path(*parts)
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,4*1024*1024)
    print('Extracted without executing code:',destination)
    return destination

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('resource',nargs='?',choices=sorted(SOURCES))
    p.add_argument('--list',action='store_true',help='List resources without network access.')
    p.add_argument('--check',action='store_true',help='Check public metadata only; do not download.')
    p.add_argument('--output-dir',type=Path,default=DATA_DIR/'downloads')
    p.add_argument('--extract',action='store_true',help='Extract ZIP to a new data/extracted/<resource> directory.')
    p.add_argument('--extract-dir',type=Path,help='Override extraction directory; must not exist.')
    p.add_argument('--archive',type=Path,help='Use an existing local revision ZIP with the pinned SHA256; no network.')
    args=p.parse_args(argv)
    if args.list or not args.resource:
        for name,s in SOURCES.items():print(name+'\t'+s['filename']+'\thttps://doi.org/'+s['doi'])
        return 0
    source=SOURCES[args.resource]
    if args.check:
        if args.archive or args.extract:p.error('--check cannot be combined with --archive or --extract')
        entry,_,_=public_file(source);print(json.dumps({'filename':source['filename'],'bytes':entry.get('size'),'checksum':entry.get('checksum')},indent=2));return 0
    if args.archive:
        if args.resource!='revision':p.error('--archive is limited to the pinned revision archive')
        archive=args.archive.resolve();verify_file(archive,source);print('Verified local archive:',archive)
    else:archive=download(source,args.output_dir.resolve())
    if args.extract:
        if not zipfile.is_zipfile(archive):p.error('The selected file is not a ZIP; no extraction performed')
        dest=safe_extract(archive,args.extract_dir or DATA_DIR/'extracted'/args.resource)
        if args.resource=='revision':print('Next: change into the extracted archive root and run python code/verify_release.py --full')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as e:
        print('ERROR:',e,file=sys.stderr);raise SystemExit(1)
