## Release

> Inspired by @NicolasHug and [torchvision](https://github.com/pytorch/vision/tree/gh-pages)

### Adding docs for a new release

Official docs for a release are built and pushed when a final tag
is created via GitHub. Docs **will not** build and push for release candidate tags.

After the final tag is created, an action will move all the docs from `main` to a directory
named after your release version. So if the release branch is called `release/0.1`, then there is a directory
on the `gh-pages` branch called `0.1/`.

### Updating `stable` versions

The stable directory is a symlink to the latest released version, and can be recreated via:

```
rm stable
ln -s 0.1 stable   # substitute the correct version number here
```

### Adding version to dropdown

In addition to updating stable, you need to update the dropdown to include
the latest version of docs.

In `versions.html`, add this line in the list right under main (substituting the correct version number):
```
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">v0.1.0 (stable)</a>
</li>
```

Remove `(stable)` from the previous version in the list.

### Push commit

Push changes to this branch `gh-pages` on main.

```
git add stable
git add version.html
git commit -m "Update stable to 0.1"
git push -u origin
```
